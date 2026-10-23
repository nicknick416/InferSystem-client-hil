"""Pure per-tick policy command limits; human actions bypass these limits."""
import numpy as np
from Inference.action_processing import rotation6d_to_quaternion
from Inference.dispatch import _clamp_quaternion_step, quaternion_to_rotation6d

class PolicyMotionLimiter:
    def __init__(self,config,fps,gripper_scale):
        self.config=dict(config);self.fps=float(fps);self.scale=float(gripper_scale);self.last=None
        self.enabled=self.config.get('enabled',False)
        if not isinstance(self.enabled,bool):raise ValueError('policy_motion_limits.enabled must be boolean')
        for key,default in [('linear_speed_m_s',.03),('angular_speed_rad_s',.2),('gripper_speed_m_s',.02)]:
            value=float(self.config.get(key,default))
            if not np.isfinite(value) or value<=0:raise ValueError('Invalid policy motion speed: '+key)
            self.config[key]=value
    def reset(self):self.last=None
    def commit(self,command):self.last=np.asarray(command,dtype=float).copy()
    @staticmethod
    def rotation(x):
        a=x[:3];a=a/np.linalg.norm(a);b=x[3:6]-np.dot(x[3:6],a)*a;b=b/np.linalg.norm(b)
        return np.stack((a,b,np.cross(a,b)),axis=1)
    def preview(self,requested,state):
        action=np.asarray(requested,dtype=float).copy()
        if not self.enabled:return action.tolist()
        if not np.isfinite(action).all():raise ValueError('Nonfinite policy action')
        ref=self.last.copy() if self.last is not None else np.asarray(state,dtype=float).copy()
        if self.last is None:ref[9]*=self.scale
        delta=action[:3]-ref[:3];limit=self.config['linear_speed_m_s']/self.fps
        action[:3]=ref[:3]+delta*min(1.,limit/max(np.linalg.norm(delta),1e-12))
        q0=rotation6d_to_quaternion(ref[3:9]);q1=rotation6d_to_quaternion(action[3:9])
        q=_clamp_quaternion_step(q0,q1,self.config['angular_speed_rad_s']/self.fps)
        action[3:9]=quaternion_to_rotation6d(q)
        limit=self.config['gripper_speed_m_s']/self.fps*self.scale
        action[9]=ref[9]+np.clip(action[9]-ref[9],-limit,limit)
        return action.tolist()
