"""Compress confirmed holds before writing images or numeric rows, in both control modes."""
from math import ceil, cos, isfinite, radians

import numpy as np


def enabled(hil):
    value = hil.get('omit_stationary_frames', False)
    if type(value) is not bool:
        raise ValueError('hil.omit_stationary_frames must be boolean')
    if value and (hil.get('recording', {}).get('format') != 'collector_h265'
                  or hil.get('omit_settled_human_frames') or hil.get('trim_terminal_policy_hold')
                  or hil.get('server_capture') or hil.get('expo_online')
                  or hil.get('keyboard_hil', {}).get('enabled')):
        raise ValueError('Use one idle filter and local SpaceMouse collector_h265 recording')
    return value


def rotation_close(a, b, deg):
    u,v=a[3:6],a[6:9]; x,y=b[3:6],b[6:9]
    ux,vy=u@x,v@y
    return ux+vy+ux*vy-(u@y)*(v@x) >= 1+2*cos(radians(deg))-1e-10


def sample(row):
    try:
        if row['mode'] not in ('policy','human') or row['segment'] is None:
            return None
        if row['mode']=='human':
            axes=np.asarray(row['spacemouse_axes'],dtype=float)
            if axes.shape!=(6,) or np.any(axes) or type(row['spacemouse_buttons']) is not int or row['spacemouse_buttons']:
                return None
        s=np.asarray(row['state'][:10],dtype=float); a=np.asarray(row['published_action'][:10],dtype=float)
        j=np.asarray(row['robot_state']['joint_positions'],dtype=float)
        t=float(row['state_time']); age=float(row['publish_start'])-t
        if (s.shape!=(10,) or a.shape!=(10,) or j.shape!=(6,) or not isfinite(t) or not 0<=age<=.1
            or not all(np.isfinite(x).all() for x in (s,a,j))): return None
        for x in (s,a):
            u,v=x[3:6],x[6:9]
            if max(abs(u@u-1),abs(v@v-1),abs(u@v))>1e-5: return None
        if (np.linalg.norm(s[:3]-a[:3])>.003 or not rotation_close(s,a,1.)
            or abs(s[9]-.085*a[9])>.004): return None  # still pursuing a target
        return (s,a,j,t,row['mode'],row['segment'])
    except (KeyError,TypeError,ValueError,OverflowError,IndexError):
        return None


def near(ref,cur,tight):
    s,a,j=ref[:3]; x,b,k=cur[:3]
    # Gripper feedback on this UR is quantized. Preserve even one width step or new target.
    return (abs(s[9]-x[9])<1.e-6 and abs(a[9]-b[9])*.085<1.e-5
            and np.linalg.norm(s[:3]-x[:3])<=(.00015 if tight else .0005)
            and np.linalg.norm(a[:3]-b[:3])<=(1.e-7 if tight else .0005)
            and rotation_close(s,x,.05 if tight else .15)
            and rotation_close(a,b,.001 if tight else .15)
            and np.max(abs(j-k))<=radians(.05 if tight else .15))


class StationaryFrames:
    """Save the first tick of a hold; buffer at most a short confirmation window."""
    def __init__(self,writer,fps):
        self.writer,self.fps=writer,float(fps)
        self.window=ceil(.30*self.fps)
        self.seen=0; self.last_kept=-1; self.gaps=[]; self.buffer=[]
        self.anchor=None; self.previous_time=None; self.confirmed=False; self.tight=True

    def keep(self,row):
        self.writer.add(dict(row,frame_index=self.writer.tick_count)); self.last_kept=row['tick']

    def drop(self,row):
        i=row['tick']
        if not self.gaps or self.gaps[-1]['last_tick']+1!=i:
            self.gaps.append(dict(first_tick=i,anchor_tick=self.last_kept,mode=row['mode'],
                                  segment=row['segment'],first_time=row['time']))
        self.gaps[-1].update(last_tick=i,last_time=row['time'],last_state_time=row['state_time'])

    def flush(self,drop):
        for r in self.buffer:
            if r['kind']=='tick': (self.drop if drop else self.keep)(r)
            else: self.writer.add(r)
        self.buffer.clear()

    def add(self,row):
        if row['kind']!='tick':
            if row['kind']=='episode_end':
                self.flush(self.tight); self.anchor=None
                if row.get('last_tick')!=self.seen-1:
                    raise ValueError('End marker does not match last physical control tick')
                row=dict(row,source_last_tick=row['last_tick'],last_tick=self.last_kept)
            if self.buffer:self.buffer.append(row)
            else:self.writer.add(row)
            return
        i=row['tick']
        if i!=self.seen:raise ValueError('Source ticks are not consecutive before idle filtering')
        self.seen+=1; cur=sample(row)
        if (cur is not None and self.anchor is not None and cur[4:]==self.anchor[4:]
            and 0<cur[3]-self.previous_time<=2/self.fps and near(self.anchor,cur,False)):
            if (not self.confirmed and self.tight and self.buffer and
                np.linalg.norm(self.anchor[1][:3]-cur[1][:3])>1.e-7):
                # A new target ends even a short fixed-command pause; keep its first command.
                self.flush(True);self.keep(row);self.anchor=None
                return
            # A small, consistent displacement over several ticks is slow travel, not jitter.
            components=(cur[0][:3],cur[1][:3],cur[2],cur[0][3:9],cur[1][3:9])
            origins=(self.anchor[0][:3],self.anchor[1][:3],self.anchor[2],self.anchor[0][3:9],self.anchor[1][3:9])
            self.travel += [np.linalg.norm(v-p) for v,p in zip(components,self.previous)]
            drift=any(np.linalg.norm(v-o)>max(limit,.78*d) for v,o,d,limit in
                      zip(components,origins,self.travel,(.00012,.00012,radians(.045),.001,.001)))
            if drift and i-self.anchor_tick>=2:
                self.flush(False);self.keep(row);self.anchor=None
                return
            self.previous=components;self.previous_time=cur[3]
            if self.confirmed:self.drop(row);return
            self.buffer.append(row);self.tight=self.tight and near(self.anchor,cur,True)
            if i-self.anchor_tick>=self.window and cur[3]-self.anchor[3]>=.30-1e-7:
                self.flush(True);self.confirmed=True
            elif i-self.anchor_tick>2*self.window:
                self.flush(False);self.anchor=None
            return
        self.flush(self.tight);self.keep(row)
        self.anchor=cur;self.anchor_tick=i;self.previous_time=cur[3] if cur is not None else None
        self.confirmed=False;self.tight=True
        if cur is not None:
            self.previous=(cur[0][:3],cur[1][:3],cur[2],cur[0][3:9],cur[1][3:9])
            self.travel=np.zeros(5)

    def finish(self):
        self.flush(self.tight)
        counts={m:sum(g['last_tick']-g['first_tick']+1 for g in self.gaps if g['mode']==m) for m in ('human','policy')}
        total=sum(counts.values())
        if self.seen!=self.writer.tick_count+total:raise ValueError('Unaccounted source frames')
        tail=self.gaps[-1] if self.gaps and self.gaps[-1]['last_tick']==self.seen-1 else None
        return dict(source_tick_count=self.seen,source_terminal_last_tick=self.seen-1,terminal_last_tick=self.last_kept,
                    omitted_stationary_frames=total,omitted_stationary_ranges=self.gaps,
                    omitted_human_idle_frames=counts['human'],omitted_policy_idle_frames=counts['policy'],
                    omitted_policy_tail_frames=(tail['last_tick']-tail['first_tick']+1) if tail and tail['mode']=='policy' else 0)
