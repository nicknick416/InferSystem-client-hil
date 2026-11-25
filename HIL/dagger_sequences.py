"""Time compression in canonical UR coordinates; all returned indices refer to saved rows."""
import math
import numpy as np


def _rot(a,b):
    # The canonical rotation is the first two columns of the rotation matrix.
    u,v=a[3:6],a[6:9]; x,y=b[3:6],b[6:9]
    return math.degrees(math.acos(float(np.clip((u@x+v@y+np.cross(u,v)@np.cross(x,y)-1)/2,-1,1))))


def _close(a,b,xyz,deg,grip):
    return (np.linalg.norm(a[:3]-b[:3]) <= xyz and abs(a[9]-b[9]) <= grip
            and _rot(a,b) <= deg)


def arrays(rows):
    s=np.asarray([r['state'] for r in rows],dtype=float)
    a=np.asarray([r['published_action'] for r in rows],dtype=float)
    j=np.asarray([r['robot_state']['joint_positions'] for r in rows],dtype=float)
    if s.shape!=(len(rows),32) or a.shape!=s.shape or j.shape!=(len(rows),6) or not all(np.isfinite(v).all() for v in (s,a,j)):
        raise ValueError('Invalid UR state, command or joint feedback')
    return s,a,j


def policy_tail(rows, fps):
    """Keep the first frame of a settled suffix. A resumed motion is never part of the suffix."""
    if not rows or rows[-1]['mode']!='policy': return len(rows)
    s,a,j=arrays(rows); last=len(rows)-1; first=last+1
    for i in range(last,-1,-1):
        r=rows[i]; age=r['publish_start']-r['state_time']
        if (r['mode']!='policy' or r['segment']!=rows[-1]['segment'] or not 0<=age<=.1
                or not _close(s[i],s[last],.0005,.15,.0005)
                or not _close(a[i],a[last],.001,.3,.001/.085)
                or np.max(abs(j[i]-j[last]))>math.radians(.15)
                or np.linalg.norm(a[i,:3]-s[i,:3])>.003 or _rot(s[i],a[i])>1
                or abs(a[i,9]*.085-s[i,9])>.004
                or (i<last and (rows[i+1]['tick']!=r['tick']+1 or not 0<rows[i+1]['state_time']-r['state_time']<=2/fps))):break
        first=i
    if last-first < math.ceil(.30*fps) or rows[last]['state_time']-rows[first]['state_time']<.30:return len(rows)
    return first+1


def segments(rows, omitted, human_only, stationary_ranges=(), fps=30):
    """Join certified static gaps only. Real interventions are never bridged in human-only data."""
    if not rows: return []
    # The writer checked every skipped tick against the retained anchor. The next
    # actual tick may legitimately start moving or change ownership.
    certified={(g['first_tick'],g['last_tick']):g for g in stationary_ranges}
    s,a,j=arrays(rows); blocks=[]; block=[]; prior=None
    for i,r in enumerate(rows):
        if human_only and r['mode']!='human':
            if block:blocks.append(block);block=[]
            prior=None;continue
        if r['mode']=='human' and not np.array_equal(r['human_action'],a[i]):
            raise ValueError(f"Human and executed labels differ at {r['tick']}")
        if prior is not None:
            p=rows[prior]; gap=r['tick']-p['tick']; okay=gap==1 and 0<r['state_time']-p['state_time']<.067
            if gap>1:
                okay=((p['tick']+1,r['tick']-1) in omitted and p['mode']==r['mode']=='human'
                      and r['state_time']>p['state_time'] and p['segment']==r['segment'] and _close(s[prior],s[i],.0015,.4,.001)
                      and _close(a[prior],a[i],.0015,.4,.001/.085)
                      and np.max(abs(j[i]-j[prior]))<=math.radians(.4))
                certificate=certified.get((p['tick']+1,r['tick']-1))
                if certificate is not None:
                    okay=(certificate['anchor_tick']==p['tick'] and certificate['mode']==p['mode']
                          and certificate['segment']==p['segment'] and p['state_time']<certificate['last_state_time']
                          and 0<r['state_time']-certificate['last_state_time']<=2/fps)
            # Adjacent real control ticks are original data, including takeovers and
            # real gripper setpoint changes. No synthetic action or mode-based split.
            if not okay or (human_only and p['segment']!=r['segment']):
                blocks.append(block);block=[]
        block.append(i);prior=i
    if block:blocks.append(block)
    return blocks
