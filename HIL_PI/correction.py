"""Recovery correction capture, durable async upload, and bounded response validation."""
import base64,json,logging,math,os,queue,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
from HIL_PI.correction_geometry import relative,pose

LOG=logging.getLogger('hil')
DEFAULTS=dict(action_mode='se3',enabled=False,auto_execute=False,server='http://127.0.0.1:5558',
    token_file='Config/correction.token',data_dir='correction_data',wrist_key='observation/wrist_image',
    max_translation_m=.02,max_rotation_rad=.15,request_timeout_s=3.,max_anchor_drift_m=.001,
    max_anchor_rotation_rad=.01,max_auto_per_episode=1,motion_mode='inference',
    linear_speed_m_s=.005,angular_speed_rad_s=.1,resume_delay_s=.1)


def config(value=None):
    value={} if value is None else value
    if not isinstance(value,dict) or set(value)-set(DEFAULTS):raise ValueError('invalid hil.correction keys')
    c=dict(DEFAULTS,**value)
    for k in ('enabled','auto_execute'):
        if type(c[k]) is not bool:raise ValueError('correction.'+k+' must be bool')
    for k in ('max_translation_m','max_rotation_rad','request_timeout_s','max_anchor_drift_m','max_anchor_rotation_rad','linear_speed_m_s','angular_speed_rad_s','resume_delay_s'):
        if isinstance(c[k],bool) or not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<=0:raise ValueError('invalid correction.'+k)
    if c['action_mode'] not in ('se3','xyz'):raise ValueError('invalid correction.action_mode')
    if c['motion_mode'] not in ('limited','inference'):raise ValueError('invalid correction.motion_mode')
    if type(c['max_auto_per_episode']) is not int or c['max_auto_per_episode']<1:raise ValueError('invalid max_auto_per_episode')
    if c['auto_execute'] and not c['enabled']:raise ValueError('auto_execute requires enabled')
    return c


def atomic(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


class CorrectionBridge:
    def __init__(self,cfg,root):
        self.cfg=config(cfg);self.root=Path(root)
        self.data=self.root/self.cfg['data_dir'];self.data.mkdir(parents=True,exist_ok=True)
        for name in ('outbox','sent','rejected'):(self.data/name).mkdir(exist_ok=True)
        self.token=(self.root/self.cfg['token_file']).read_text().strip()
        self.queue=queue.Queue(maxsize=64);self.stop=threading.Event()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='correction-predict')
        self.future=None;self.future_cycle=None;self.anchor=None;self.human_seen=False;self.auto_count=0
        self.writer=threading.Thread(target=self.run,name='correction-outbox',daemon=True);self.writer.start()

    def request(self,path,payload):
        req=Request(self.cfg['server']+path,data=json.dumps(payload,allow_nan=False).encode(),
                    headers={'Content-Type':'application/json','Authorization':'Bearer '+self.token})
        with urlopen(req,timeout=self.cfg['request_timeout_s']) as response:
            return json.loads(response.read(5_000_000))

    def audit(self,event,**fields):
        self.queue.put_nowait(dict(audit=True,event=event,wall_time=time.time(),**fields))

    def new_episode(self):
        self.clear();self.auto_count=0

    def clear(self):
        self.anchor=None;self.human_seen=False;self.future_cycle=None

    def capture(self,state,payload,stamp,sensor_times,published,episode_id,tick):
        image=payload.get(self.cfg['wrist_key'])
        if not isinstance(image,bytes):
            LOG.error('[CORRECTION] no encoded wrist image; capture disabled for this cycle');self.clear();return None
        pose(state[:9])
        self.anchor=dict(schema=1,sample_id=uuid.uuid4().hex,anchor_pose=list(state[:9]),
            anchor_time=stamp,anchor_tick=tick,episode_id=episode_id,anchor_rgb_time=sensor_times.get('rgb_read_complete'),
            image=image,gripper_command=float(published[9]))
        self.human_seen=False
        self.audit('anchor_captured',sample_id=self.anchor['sample_id'],episode_id=episode_id,tick=tick)
        LOG.info('[CORRECTION] anchor captured id=%s',self.anchor['sample_id'])
        return self.anchor['sample_id']

    def human(self):
        if self.anchor is not None:
            if not self.human_seen:self.audit('human_seen',sample_id=self.anchor['sample_id'])
            self.human_seen=True
        self.future_cycle=None  # A late prediction must never override human takeover.

    def resume(self,state,payload,stamp,sensor_times,tick):
        if self.anchor is None or not self.human_seen:
            reason='missing_anchor' if self.anchor is None else 'no_human_intervention'
            result=dict(status='skipped',reason=reason,tick=tick)
            self.audit('resume_skipped',**result)
            LOG.warning('[CORRECTION] p sample skipped: %s tick=%s',reason,tick)
            self.clear()
            return result
        a=self.anchor;sample={k:v for k,v in a.items() if k not in ('image','gripper_command')}
        sample.update(action_mode=self.cfg['action_mode'],supervision='human_resume',target_pose=list(state[:9]),resume_time=stamp,resume_tick=tick,
                      resume_rgb_time=sensor_times.get('rgb_read_complete'))
        error=None
        try:
            target=sample['target_pose']
            if self.cfg['action_mode']=='xyz':
                pose(target)  # Validate full recorded pose, supervise translation only.
                target=[*target[:3],*a['anchor_pose'][3:9]]
            relative(a['anchor_pose'],target,self.cfg['max_translation_m'],self.cfg['max_rotation_rad'])
        except ValueError as exc:error=str(exc)
        # Encoding and all disk/network writes run outside the control loop.
        self.queue.put_nowait((sample,a['image'],payload.get(self.cfg['wrist_key']),error))
        self.audit('resume_queued',sample_id=sample['sample_id'],tick=tick,rejection=error)
        self.clear()
        LOG.info('[CORRECTION] human target queued id=%s accepted=%s',sample['sample_id'],error is None)
        return dict(sample_id=sample['sample_id'],status='rejected' if error else 'queued',reason=error)

    def predict(self):
        if not self.cfg['auto_execute'] or self.anchor is None or self.human_seen or self.auto_count>=self.cfg['max_auto_per_episode']:return False
        if self.future is not None:return False
        a=dict(self.anchor);cycle=a['sample_id'];self.future_cycle=cycle;self.auto_count+=1
        def call():
            return self.request('/predict',dict(action_mode=self.cfg['action_mode'],cycle_id=cycle,anchor_pose=a['anchor_pose'],
                wrist_png=base64.b64encode(a['image']).decode(),max_translation_m=self.cfg['max_translation_m'],max_rotation_rad=self.cfg['max_rotation_rad']))
        self.future=self.pool.submit(call);return True

    def poll(self,state):
        if self.future is None or not self.future.done():return None
        future=self.future;self.future=None
        try:result=future.result()
        except Exception as exc:LOG.warning('[CORRECTION] prediction failed; manual hold: %s',exc);return None
        if self.anchor is None or self.human_seen or self.future_cycle!=self.anchor['sample_id']:return None
        self.future_cycle=None
        if result.get('status')!='ok':LOG.info('[CORRECTION] not ready: %s',result);return None
        try:
            if result.get('action_mode','se3')!=self.cfg['action_mode']:raise ValueError('action mode mismatch')
            if result.get('cycle_id')!=self.anchor['sample_id']:raise ValueError('cycle mismatch')
            if result.get('bounds')!=dict(max_translation_m=self.cfg['max_translation_m'],max_rotation_rad=self.cfg['max_rotation_rad']):raise ValueError('bounds mismatch')
            candidates=result['poses'];scores=np.asarray(result['scores'],float);index=result['selected']
            if not 1<=len(candidates)<=128 or scores.shape!=(len(candidates),) or not np.isfinite(scores).all():raise ValueError('invalid scores/candidates')
            if type(index) is not int or index!=int(np.argmax(scores)):raise ValueError('selection is not Q argmax')
            for target in candidates:
                relative(self.anchor['anchor_pose'],target,self.cfg['max_translation_m'],self.cfg['max_rotation_rad'])
                if self.cfg['action_mode']=='xyz' and not np.allclose(target[3:9],self.anchor['anchor_pose'][3:9],rtol=0,atol=1e-7):
                    raise ValueError('XYZ candidate changed anchor rotation')
            relative(self.anchor['anchor_pose'],state[:9],self.cfg['max_anchor_drift_m'],self.cfg['max_anchor_rotation_rad'])
            relative(state[:9],candidates[index],self.cfg['max_translation_m'],self.cfg['max_rotation_rad'])
        except (ValueError,KeyError,TypeError) as exc:LOG.error('[CORRECTION] rejected prediction: %s',exc);return None
        return result

    def run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:item=self.queue.get(timeout=.2)
            except queue.Empty:item=None
            if isinstance(item,dict) and item.get('audit'):
                try:
                    with (self.data/'capture_audit.jsonl').open('a') as stream:
                        stream.write(json.dumps(item,allow_nan=False)+'\n')
                except Exception:LOG.exception('[CORRECTION] audit persistence failed')
                continue
            if item is not None:
                sample,image,target_image,error=item
                sample['wrist_png']=base64.b64encode(image).decode()
                if isinstance(target_image,bytes):sample['target_wrist_png']=base64.b64encode(target_image).decode()
                if error:sample['rejection']=error
                folder='rejected' if error else 'outbox'
                try:atomic(self.data/folder/(sample['sample_id']+'.json'),sample)
                except Exception:LOG.exception('[CORRECTION] sample persistence failed')
            if self.stop.is_set():continue
            for path in sorted((self.data/'outbox').glob('*.json')):
                try:
                    result=self.request('/sample',json.loads(path.read_text()))
                    if result.get('status')!='ok':raise RuntimeError(str(result))
                    os.replace(path,self.data/'sent'/path.name)
                    LOG.info('[CORRECTION] uploaded %s',path.stem)
                except HTTPError as exc:
                    if exc.code==422:
                        LOG.error('[CORRECTION] server rejected %s: %s',path.stem,exc.read(1000))
                        os.replace(path,self.data/'rejected'/path.name)
                    else:LOG.warning('[CORRECTION] upload pending: HTTP %d',exc.code);break
                except Exception as exc:LOG.warning('[CORRECTION] upload pending: %s',exc);break
            self.stop.wait(.8)

    def close(self):
        self.stop.set();self.writer.join(15);self.pool.shutdown(wait=True,cancel_futures=True)
        if self.writer.is_alive():raise RuntimeError('correction writer failed to stop')
