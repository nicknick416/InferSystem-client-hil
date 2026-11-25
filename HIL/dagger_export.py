"""Session-local v2.1 datasets for N0-VTLA. No model or robot dependency."""
import argparse
import fcntl
import itertools
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from HIL.dagger_sequences import segments,policy_tail
from HIL.dagger_storage import video_info,video_slice
from HIL.human_idle_recording import SettledHumanFrames
from HIL.video_storage import iter_records

CAMS={'observation/image':'third_view','observation/wrist_image':'left_wrist_view',
      'observation/left_tactile':'left_wrist_left_tactile','observation/right_tactile':'left_wrist_right_tactile'}
DP='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
VP='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'


def stat(x):
    return {k:np.atleast_1d(v).tolist() for k,v in [('min',x.min(0)),('max',x.max(0)),('mean',x.mean(0)),('std',x.std(0)),('count',len(x))]}


def selected_video(source,dest,ids,meta):
    fps=meta['fps'];cfg=meta['config']['hil']['recording'];n,w,h,codec,_,_=video_info(source)
    if any(y<=x for x,y in zip(ids,ids[1:])) or ids[0]<0 or ids[-1]>=n:raise ValueError('Unaligned camera refs')
    if ids[-1]-ids[0]+1==len(ids):video_slice(source,dest,ids[0],ids[-1],n,fps,cfg)
    else:
        from HIL.video_storage import find_ffmpeg
        import subprocess
        ranges=[list(g) for _,g in itertools.groupby(enumerate(ids),key=lambda x:x[1]-x[0])];sel='+'.join(f'between(n,{g[0][1]},{g[-1][1]})' for g in ranges)
        subprocess.run([find_ffmpeg(cfg.get('ffmpeg')),'-nostdin','-v','error','-vaapi_device',cfg.get('vaapi_device','/dev/dri/renderD128'),'-threads','1','-i',str(source),'-vf',f"select='{sel}',setpts=N/({fps}*TB),format=nv12,hwupload",'-an','-c:v','hevc_vaapi','-qp',str(cfg.get('qp',30)),'-y',str(dest)],check=True,timeout=300)
    if video_info(dest)[:4]!=(len(ids),w,h,codec):raise ValueError('Encoded stream does not match numeric trajectory')
    return {'dtype':'video','shape':[h,w,3],'names':['height','width','channels'],'info':{'video.height':h,'video.width':w,'video.channels':3,'video.fps':fps,'video.codec':codec,'video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}}


def camera_stats(path):
    # Per-channel statistics on up to 12 evenly spaced thumbnail frames (RGB, [0,1]).
    with av.open(str(path)) as f:
        n=f.streams.video[0].frames; wanted=set(np.linspace(0,n-1,min(n,12),dtype=int));samples=[]
        for i,frame in enumerate(f.decode(video=0)):
            if i in wanted:samples.append(frame.reformat(width=64,height=48,format='rgb24').to_ndarray())
        if i+1!=n or len(samples)!=len(wanted):raise ValueError('Video decode count mismatch')
    rgb=np.asarray(samples,dtype=np.float32).reshape(-1,3)/255; st=stat(rgb);st['count']=[len(samples)]
    return {k:(v if k=='count' else np.array(v).reshape(3,1,1).tolist()) for k,v in st.items()}


def save_baselines(src, root, row, uid):
    from PIL import Image
    refs = row['observation'].get('extra', {}).get('tactile_baseline') or {}
    result = {}
    if not uid or Path(uid).name != uid: raise ValueError('Invalid source episode id')
    for key in ('observation/left_tactile','observation/right_tactile'):
        if key not in refs: continue
        ref=refs[key];stream=ref['stream']
        if Path(stream).name!=stream or not stream.startswith('baseline.'): raise ValueError('Invalid baseline stream')
        rel=Path('baselines')/uid/(CAMS[key]+'.png');dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            with av.open(str(src/stream/'video.mp4')) as f:
                img=next(frame.to_image() for n,frame in enumerate(f.decode(video=0)) if n==ref['frame'])
            temp=dst.with_suffix('.tmp');img.save(temp,format='PNG');os.replace(temp,dst)
        result['observation.image.'+CAMS[key]]=str(rel)
    return result


def dataset_view(root,src,meta,rows,blocks,view):
    if not blocks:return 0
    root.mkdir(parents=True,exist_ok=True);md=root/'meta';version=root/'.versions';version.mkdir(exist_ok=True)
    def lines(name):return [json.loads(l) for l in (md/name).read_text().splitlines()] if md.exists() else []
    eps=lines('episodes.jsonl');stats=lines('episodes_stats.jsonl'); tasks=lines('tasks.jsonl');uid=meta['episode_id']
    baselines=save_baselines(src,root,rows[0],uid)
    existing=[e for e in eps if e['source_episode_id']==uid]
    if existing:
        if any(e.get('source_tactile_baselines')!=baselines for e in existing):
            for e in existing:e['source_tactile_baselines']=baselines
            tmp=md/'episodes.tmp';tmp.write_text(''.join(json.dumps(e)+'\n' for e in eps));os.replace(tmp,md/'episodes.jsonl')
        return 0
    task=meta['prompt'].strip().rstrip('.');task_ids={r['task']:r['task_index'] for r in tasks};ti=task_ids.get(task,len(tasks));fps=meta['fps'];start=sum(e['length'] for e in eps);old=len(eps)
    if ti==len(tasks):tasks.append(dict(task=task,task_index=ti))
    features=json.loads((md/'info.json').read_text())['features'] if md.exists() else {};stage=Path(tempfile.mkdtemp(dir=root,prefix='.build-'))
    def staged(rel):p=stage/rel;p.parent.mkdir(exist_ok=True,parents=True);return p
    try:
        for ids in blocks:
            ei=len(eps);n=len(ids);subs=[rows[i] for i in ids];ix={'episode_chunk':ei//1000,'episode_index':ei}; epstats={};keys=[]
            for k,suffix in CAMS.items():
                vk='observation.image.'+suffix;keys.append(vk);refs=[r['observation'][k] for r in subs];stream=refs[0]['stream']
                if any(v['stream']!=stream for v in refs):raise ValueError('Camera stream changed within episode')
                if Path(stream).name!=stream:raise ValueError('Invalid camera stream')
                target=staged(VP.format(**ix,video_key=vk));spec=selected_video(src/stream/'video.mp4',target,[v['frame'] for v in refs],meta)
                if vk in features and features[vk]!=spec:raise ValueError('Camera schema changed in session')
                features[vk]=spec;epstats[vk]=camera_stats(target)
            arr={'observation.state':np.array([r['state'] for r in subs],np.float32),'action':np.array([r['published_action'] for r in subs],np.float32),'action_mask':np.tile(np.arange(32)<10,(n,1)), 'timestamp':np.arange(n,dtype=np.float32)/np.float32(fps),'frame_index':np.arange(n,dtype=np.int64),'episode_index':np.full(n,ei,np.int64),'index':np.arange(start,start+n,dtype=np.int64),'task_index':np.full(n,ti,np.int64),'source_tick':np.array([r['tick'] for r in subs],np.int64),'is_human':np.array([r['mode']=='human' for r in subs],bool)}
            if view=='human_only':arr['action_mask'][:,9]=[not r['_mask_grip'] for r in subs]
            cols={};hf={}
            for k,x in arr.items():
                dtype=str(x.dtype);features[k]=dict(dtype=dtype,shape=[32 if x.ndim==2 else 1],names=None);typ=pa.from_numpy_dtype(x.dtype)
                cols[k]=pa.FixedSizeListArray.from_arrays(pa.array(x.flatten(),type=typ),32) if x.ndim==2 else pa.array(x,type=typ);val={'dtype':dtype,'_type':'Value'};hf[k]={'feature':val,'length':32,'_type':'Sequence'} if x.ndim==2 else val;epstats[k]=stat(x)
            table=pa.table(cols).replace_schema_metadata({b'huggingface':json.dumps({'info':{'features':hf}}).encode()});pq.write_table(table,staged(DP.format(**ix)));start+=n
            eps.append(dict(episode_index=ei,length=n,tasks=[task],video_keys=keys,source_session=meta.get('session_name',src.parent.name),source_folder=src.name,source_tactile_baselines=baselines,source_episode_id=uid,source_tick_first=subs[0]['tick'],source_tick_last=subs[-1]['tick'],outcome=meta['outcome']));stats.append(dict(episode_index=ei,stats=epstats))
        info=dict(codebase_version='v2.1',robot_type='single_arm_tactile',fps=fps,total_episodes=len(eps),total_frames=start,total_tasks=len(tasks),chunks_size=1000,total_chunks=(len(eps)+999)//1000,total_videos=len(eps)*len(CAMS),data_path=DP,video_path=VP,splits={'train':f'0:{len(eps)}'},features=features)
        # New data files first. Publish their mutually consistent metadata with one symlink swap.
        for file in stage.glob('**/*'):
            if file.is_file():dst=root/file.relative_to(stage);dst.parent.mkdir(parents=True,exist_ok=True);os.replace(file,dst)
        rev=version/uuid.uuid4().hex;rev.mkdir();(rev/'info.json').write_text(json.dumps(info));
        for file,items in [('episodes.jsonl',eps),('tasks.jsonl',tasks),('episodes_stats.jsonl',stats)]: (rev/file).write_text(''.join(json.dumps(v)+'\n' for v in items))
        prev=md.resolve() if md.is_symlink() else None;link=stage/'head';link.symlink_to(rev.relative_to(root));os.replace(link,md)
        if prev and prev.parent==version.resolve():shutil.rmtree(prev,ignore_errors=True)
        return len(eps)-old
    finally:shutil.rmtree(stage,ignore_errors=True)


def export_episode(path,target=None):
    path=Path(path);m=json.loads((path/'metadata.json').read_text());root=Path(target) if target else path.parent/'training';
    if m.get('status') != 'kept':
        return {'reason': 'not_kept'}
    if m.get('outcome') not in ('success', 'failure'):
        raise ValueError('A kept training episode must be labeled success or failure')
    rows=[r for r in iter_records(path) if r['kind']=='tick'];rows=rows[:policy_tail(rows,m['fps'])] if 'omitted_policy_tail_frames' not in m else rows
    omitted={(g['first_tick'],g['last_tick']) for g in m.get('omitted_human_idle_ranges',[])}
    if 'omitted_human_idle_ranges' not in m and 'omitted_stationary_ranges' not in m:
        class Sink:
            def __init__(self):self.rows=[]
            @property
            def tick_count(self):return len(self.rows)
            def add(self,r):self.rows.append(r)
        sink=Sink();gate=SettledHumanFrames(sink,m['fps']);
        for r in rows:gate.add(r)
        res=gate.finish();rows=sink.rows;omitted={(g['first_tick'],g['last_tick']) for g in res['omitted_human_idle_ranges']}
    mask=False;prior=None
    for r in rows:
        if r['mode']!='human':mask=False
        elif prior is not None and prior['mode']!='human':mask=abs(prior['published_action'][9]-r['published_action'][9])>.0301
        if r.get('spacemouse_buttons',0):mask=False
        r['_mask_grip']=mask;prior=r
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);return {view:dataset_view(root/view,path,m,rows,segments(rows,omitted,h,m.get('omitted_stationary_ranges',()),m['fps']),view) for view,h in [('full_episode',False),('human_only',True)]}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode');p.add_argument('--target');args=p.parse_args();print(export_episode(args.episode,args.target))
