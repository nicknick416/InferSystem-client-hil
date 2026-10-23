"""Small-ROI optical-flow slip evidence, evaluated only in the tactile process."""
import cv2
import numpy as np


class TactileSlip:
    def __init__(self, config):
        self.c=config
        self.reset()

    def reset(self):
        self.previous=None
        self.stamp=None
        self.energy=None
        self.started=None
        self.count=0

    def update(self, pixels, stamp, energy, active):
        report=dict(slipping=False,reason='disabled',speed_px_s=0.,coherence=0.,tracks=0,confirm_frames=0)
        if not self.c['slip_enabled']:
            return report
        if not active or energy<self.c['slip_min_energy']:
            self.reset()
            return dict(report,reason='inactive_or_no_contact')
        gray=cv2.cvtColor(cv2.resize(pixels,(80,45),interpolation=cv2.INTER_AREA)[5:40,12:68],cv2.COLOR_BGR2GRAY)
        if self.stamp==stamp:
            return dict(report,reason='duplicate_frame')
        prev,old_stamp,old_energy=self.previous,self.stamp,self.energy
        if old_stamp is None or not 0<stamp-old_stamp<=self.c['max_sample_gap_s']:
            self.reset()
            self.started=stamp
            prev=None
        self.previous,self.stamp,self.energy=gray,stamp,energy
        if prev is None:
            return dict(report,reason='priming')
        if energy<old_energy*self.c['slip_min_contact_ratio']:
            self.count=0
            self.started=stamp
            return dict(report,reason='contact_decreasing')
        points=cv2.goodFeaturesToTrack(prev,maxCorners=80,qualityLevel=.05,minDistance=3,blockSize=3)
        if points is None or len(points)<self.c['slip_min_tracks']:
            self.count=0
            return dict(report,reason='insufficient_texture')
        nxt,ok,_=cv2.calcOpticalFlowPyrLK(prev,gray,points,None,winSize=(11,11),maxLevel=1)
        if nxt is None:
            self.count=0
            return dict(report,reason='flow_failed')
        back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,prev,nxt,None,winSize=(11,11),maxLevel=1)
        if back is None:
            self.count=0
            return dict(report,reason='flow_failed')
        valid=(ok.ravel()!=0)&(back_ok.ravel()!=0)&(np.linalg.norm((back-points)[:,0],axis=1)<1.)
        valid &= np.isfinite(nxt[:,0]).all(axis=1)
        valid &= (nxt[:,0,0]>=0)&(nxt[:,0,0]<gray.shape[1])&(nxt[:,0,1]>=0)&(nxt[:,0,1]<gray.shape[0])
        flow=(nxt-points)[valid,0]
        report['tracks']=len(flow)
        if len(flow)<self.c['slip_min_tracks']:
            self.count=0
            return dict(report,reason='insufficient_tracks')
        velocity=flow/(stamp-old_stamp)
        speeds=np.linalg.norm(velocity,axis=1)
        moving=speeds>=self.c['slip_speed_px_s']
        if np.count_nonzero(moving)>=self.c['slip_min_tracks']:
            median=np.median(velocity[moving],axis=0)
            cosine=(velocity@median)/(np.maximum(speeds,1e-6)*max(np.linalg.norm(median),1e-6))
            coherent=moving&(cosine>.7)
            coherence=float(np.count_nonzero(coherent)/len(flow))
            speed=float(np.median(speeds[coherent])) if coherent.any() else 0.
        else:
            coherence=speed=0.
        report.update(speed_px_s=speed,coherence=coherence)
        armed=stamp-self.started>=self.c['slip_arm_s']
        suspected=armed and speed>=self.c['slip_speed_px_s'] and coherence>=self.c['slip_min_coherence']
        self.count=self.count+1 if suspected else 0
        slipping=self.count>=self.c['slip_confirm_frames']
        return dict(report,slipping=slipping,confirm_frames=self.count,
                    reason='coherent_contact_motion' if slipping else 'arming' if not armed else 'below_threshold')
