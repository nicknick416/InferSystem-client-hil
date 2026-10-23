"""Metric bounds in the anchor TCP frame; no clipping of targets or predictions."""
import numpy as np



def pose(value):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=(9,) or not np.isfinite(a).all():raise ValueError('pose must be finite xyz+rot6d (9 values)')
    x=a[3:6];y=a[6:9]
    if abs(np.linalg.norm(x)-1)>.02 or abs(np.linalg.norm(y)-1)>.02 or abs(np.dot(x,y))>.02:
        raise ValueError('invalid rotation6d')
    x=x/np.linalg.norm(x);y=y-np.dot(x,y)*x;y=y/np.linalg.norm(y)
    return a[:3].copy(),np.column_stack((x,y,np.cross(x,y)))


def relative(anchor,target,max_translation=.02,max_rotation=.15):
    p,R=pose(anchor);q,S=pose(target)
    delta=np.r_[R.T@(q-p),rotation_log(R.T@S)]
    if np.linalg.norm(delta[:3])>max_translation+1e-9 or np.linalg.norm(delta[3:])>max_rotation+1e-9:
        raise ValueError('correction outside configured generator bounds')
    return delta


def absolute(anchor,delta,max_translation=.02,max_rotation=.15):
    p,R=pose(anchor);d=np.asarray(delta,float)
    if d.shape!=(6,) or not np.isfinite(d).all():raise ValueError('invalid delta')
    if np.linalg.norm(d[:3])>max_translation+1e-9 or np.linalg.norm(d[3:])>max_rotation+1e-9:
        raise ValueError('out-of-bound generated delta; reject, never clip')
    S=R@rotation_exp(d[3:])
    return np.r_[p+R@d[:3],S[:,0],S[:,1]].tolist()


def rotation_exp(v):
    x,y,z=v;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    theta=np.linalg.norm(v)
    return np.eye(3)+np.sinc(theta/np.pi)*K+.5*np.sinc(theta/(2*np.pi))**2*(K@K)


def rotation_log(R):
    v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2
    sine=np.linalg.norm(v);cosine=np.clip((np.trace(R)-1)/2,-1,1)
    theta=np.arctan2(sine,cosine)
    if theta<1e-7:return v
    if np.pi-theta<1e-6:
        _,vectors=np.linalg.eigh((R+R.T)/2);axis=vectors[:,-1]
        if np.dot(axis,v)<0:axis=-axis
        return axis*theta
    return v*(theta/sine)
