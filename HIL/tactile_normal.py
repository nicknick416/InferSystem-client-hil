"""Persistent local one-sided normal-contact model; cold start never alarms."""
from pathlib import Path
import json
import time
import uuid
from collections import deque
from urllib.request import Request, urlopen
import cv2
import numpy as np


class LocalNormal:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['normal_data_dir'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.zero = None
        self.hold_epoch = 0
        self.recent = deque(maxlen=5)
        self.last_stamp = None
        self.samples = [json.loads(p.read_text())['local_scores'] for p in sorted(self.root.glob('sample_*.json'))]
        self.corrector_ready = False
        self.status_time = 0.
        self.last_poll = -float('inf')
        self.status_error = 'not_checked'
        from HIL.tactile_slip import TactileSlip
        self.slip=TactileSlip(config)
        self.reset()

    def reset(self):
        self.slip.reset()
        self.count = 0
        self.stuck = False

    def set_zero(self, encoded):
        self.zero = cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR)
        self.recent.clear()
        self.last_stamp = None
        self.reset()

    def poll_ready(self):
        now = time.monotonic()
        if now-self.last_poll < 2:
            return
        self.last_poll = now
        self.corrector_ready = False
        try:
            token = Path(self.config['normal_token_file']).read_text().strip()
            request = Request(self.config['normal_server']+'/status',data=b'{}',
                headers={'Content-Type':'application/json','Authorization':'Bearer '+token})
            with urlopen(request,timeout=.25) as response:
                status=json.loads(response.read(100000))
            self.corrector_ready = (status.get('ready') is True and
                status.get('data_dir') == self.config['normal_corrector_data_dir'])
            self.status_error = None if self.corrector_ready else 'corrector_not_ready_or_wrong_run'
            self.status_time = now
        except Exception as exc:
            self.status_error = type(exc).__name__

    def update(self, pixels, stamp, hold_epoch, active, opening):
        # opening intentionally ignored: release should reduce one-sided scores.
        self.poll_ready()
        ready = len(self.samples)>=self.config['normal_min_samples'] and self.corrector_ready
        report=dict(stuck=False,reason='waiting_zero',normal_samples=len(self.samples),
                    ready=ready,corrector_ready=self.corrector_ready,status_error=self.status_error)
        if self.zero is None or self.zero.shape!=pixels.shape:
            return report
        if self.last_stamp is not None and not 0 < stamp-self.last_stamp <= self.config['max_sample_gap_s']:
            self.recent.clear();self.count=0;self.slip.reset()
        if stamp==self.last_stamp:return dict(report,reason='duplicate_frame')
        self.last_stamp=stamp
        roi=cv2.resize(cv2.absdiff(pixels,self.zero),(80,45),interpolation=cv2.INTER_AREA)[5:40,12:68]
        scores=roi.reshape(5,7,8,7,3).mean(axis=(1,3,4)).reshape(-1)
        self.recent.append(scores)
        if hold_epoch!=self.hold_epoch:
            self.hold_epoch=hold_epoch
            self.slip.reset()
            if not ready:
                normal=np.median(np.stack(self.recent),axis=0)
                if float(np.mean(normal)) >= self.config['energy_min_hold_score']:
                    sample_id=uuid.uuid4().hex
                    data=dict(local_scores=normal.tolist(),capture_time=stamp,shape=[5,8],schema=1)
                    # All disk writes are on the detector process, never control thread.
                    cv2.imwrite(str(self.root/(sample_id+'_zero.png')),self.zero)
                    cv2.imwrite(str(self.root/(sample_id+'_hold.png')),pixels)
                    tmp=self.root/(sample_id+'.tmp');tmp.write_text(json.dumps(data))
                    tmp.replace(self.root/('sample_'+sample_id+'.json'))
                    self.samples.append(normal.tolist())
                else:report['sample_rejected']='insufficient_contact'
        report['normal_samples']=len(self.samples)
        if not self.samples:return dict(report,reason='cold_start_collecting',local_scores=scores.tolist())
        normal=np.asarray(self.samples)
        center=np.median(normal,axis=0)
        mad=np.median(np.abs(normal-center),axis=0)
        scale=np.maximum(center,self.config['energy_min_hold_score'])
        # Threshold exceeds ALL observed normal examples plus a positive margin.
        upper=np.maximum(normal.max(axis=0)+2*self.config['energy_margin']*scale,center+6*1.4826*mad)
        energy=.5*scores/scale
        thresholds=.5*upper/scale
        over=scores>upper
        ready=len(self.samples)>=self.config['normal_min_samples'] and self.corrector_ready
        if ready and active and np.count_nonzero(over)>=self.config['normal_min_blocks']:
            self.count+=1
            self.stuck |= self.count>=self.config['energy_confirm_frames']
        else:
            self.count=0
            self.stuck=False
        slip=self.slip.update(pixels,stamp,float(np.mean(energy)),ready and active)
        energy_stuck=self.stuck
        trigger_reasons=(['tactile_deformation'] if energy_stuck else [])+(['tactile_slip'] if slip['slipping'] else [])
        return dict(report,ready=ready,stuck=bool(trigger_reasons),slip=slip,trigger_reasons=trigger_reasons,
                    energy_stuck=energy_stuck,reason='+'.join(trigger_reasons) if trigger_reasons else
                    'cold_start_collecting' if not ready else 'local_energy_normal' if active else 'non_policy',
                    local_scores=scores.tolist(),local_energy=energy.tolist(),local_threshold=thresholds.tolist(),
                    exceeded_blocks=int(np.count_nonzero(over)),energy=float(np.mean(energy)),confirm_frames=self.count)
