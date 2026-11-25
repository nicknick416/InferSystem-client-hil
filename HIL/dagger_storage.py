"""Trim the pending local H.265 capture, publishing only a verified prefix."""
import csv
import errno
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import av
from HIL.dagger_sequences import policy_tail
from HIL.video_storage import find_ffmpeg, iter_records


def video_info(path):
    with av.open(str(path)) as f:
        v=f.streams.video[0]
        return v.frames,v.width,v.height,v.codec_context.name,bool(v.codec_context.has_b_frames),float(v.average_rate)


def video_slice(src,dst,first,last,source_count,fps,settings):
    n,w,h,codec,b,rate=video_info(src)
    if n!=source_count or not (0<=first<=last<n) or abs(rate-fps)>1e-4:raise ValueError(f'Invalid video bounds/fps: {src}')
    if first==0 and last+1==n:
        try:
            os.link(src,dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV: raise
            shutil.copyfile(src,dst)
        return
    # Preserve the original compressed bytes when clipping a decodable prefix.
    args=[find_ffmpeg(settings.get('ffmpeg')),'-nostdin','-v','error','-y']
    if first==0 and not b:
        args+=['-i',str(src),'-map','0:v:0','-frames:v',str(last+1),'-c:v','copy']
    else:
        args+=['-vaapi_device',settings.get('vaapi_device','/dev/dri/renderD128'),'-threads','1','-i',str(src),'-vf',f'trim=start_frame={first}:end_frame={last+1},setpts=PTS-STARTPTS,format=nv12,hwupload','-an','-c:v','hevc_vaapi','-qp',str(settings.get('qp',30))]
    subprocess.run([*args,str(dst)],check=True,timeout=300)
    actual=video_info(dst)
    if actual[:4]!=(last-first+1,w,h,codec) or abs(actual[-1]-fps)>1e-4:raise ValueError(f'Video verification failed: {dst}')


def prefix_file(path,fn):
    tmp=path.with_name(path.name+'.new')
    with path.open() as src,tmp.open('w') as dst:fn(src,dst)
    os.replace(tmp,path)


def trim_capture(source, dest, meta, rows, count):
    """dest is private staging on the same filesystem; never mutate a source hard link."""
    import itertools
    shutil.copytree(source,dest,copy_function=os.link)
    last=rows[count-1]['tick']; old_last=rows[-1]['tick']; end=rows[count]['time'];fps=meta['fps'];cfg=meta['config']['hil']['recording']; vids=dict(meta['video_frames'])
    def prefix(f,g):g.writelines(itertools.islice(f,count+1))
    prefix_file(dest/'hil/ticks.csv',prefix)
    def numeric(f,g):
        rd=csv.DictReader(f);wr=csv.DictWriter(g,fieldnames=rd.fieldnames);wr.writeheader()
        wr.writerows(r for r in rd if int(r['tick'])<=last)
    for p in dest.glob('*/data.csv'):prefix_file(p,numeric)
    prefix_file(dest/'hil/observations.jsonl',lambda f,g:g.writelines(itertools.islice(f,count)))
    def events(f,g):
        for line in f:
            r=json.loads(line)
            if r.get('__hil_tick_row__',-1)>=count:continue
            if r.get('kind')=='episode_end':r.update(source_last_tick=old_last,last_tick=last,source_time=r.get('time'),time=end);line=json.dumps(r)+'\n'
            g.write(line)
    prefix_file(dest/'hil/events.jsonl',events)
    for name,n in list(vids.items()):
        folder=dest/name; ts=folder/'timestamps.csv'
        with ts.open() as f: log=list(csv.DictReader(f))
        refs=[int(x['tick_first_reference']) for x in log]; size=sum(t<=last for t in refs)
        if len(log)!=n or any(t>last for t in refs[:size]):raise ValueError('Unexpected video timestamp ordering')
        if size==n:continue
        if size==0:
            shutil.rmtree(folder);del vids[name];continue
        out=folder/'new.mp4'; video_slice(folder/'video.mp4',out,0,size-1,n,fps,cfg);os.replace(out,folder/'video.mp4');vids[name]=size
        prefix_file(ts,lambda f,g:g.writelines(itertools.islice(f,size+1)))
        def config(f,g):
            rd=csv.DictReader(f);wr=csv.DictWriter(g,fieldnames=rd.fieldnames);wr.writeheader()
            for rec in rd:rec['frame_count']=size;wr.writerow(rec)
        prefix_file(folder/'config.csv',config)
    # Diagnostic events are retained on the source clock. Training uses kept tick rows only.
    return dict(total_frames=count,video_frames=vids,terminal_last_tick=last,source_terminal_last_tick=old_last,
                source_capture_end_time=meta.get('capture_end_time'),capture_end_time=end,
                omitted_policy_tail_frames=len(rows)-count,omitted_policy_tail_ticks=[rows[count]['tick'],old_last])


def prepare_tail(episode):
    rows=[r for r in iter_records(episode.path,observations=False) if r['kind']=='tick'];count=policy_tail(rows,episode.metadata['fps'])
    if count==len(rows):return None,dict(omitted_policy_tail_frames=0)
    stage=Path(tempfile.mkdtemp(dir=episode.path.parent,prefix='.tail-'))
    try:return stage,trim_capture(episode.path,stage/'capture',episode.metadata,rows,count)
    except BaseException:
        shutil.rmtree(stage);raise
