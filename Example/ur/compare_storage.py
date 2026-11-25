"""Convert one old episode non-destructively and measure H.265 quality/size."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from HIL.recording import records
from HIL.video_storage import (CollectorWriter, VideoDecoder, iter_records, probe,
                               VIDEO_KEY, BYTES_KEY, is_image, FORMAT)


def skeleton(value):
    if isinstance(value, bytes):
        return {'image_hash':hashlib.sha256(value).hexdigest()} if is_image(value) else value
    if isinstance(value, dict):
        if VIDEO_KEY in value:
            return {'image_hash':value['source_sha256']}
        if BYTES_KEY in value:
            import base64
            return base64.b64decode(value[BYTES_KEY])
        return {k:skeleton(v) for k,v in value.items()}
    if isinstance(value, list):
        return [skeleton(v) for v in value]
    return value


def quality(a, b):
    delta = a.astype(np.float64)-b
    mse = float(np.mean(delta**2))
    x, y = a.astype(np.float64), b.astype(np.float64)
    blur = lambda v:cv2.GaussianBlur(v, (11,11), 1.5)
    mx, my = blur(x), blur(y)
    vx, vy, vxy = blur(x*x)-mx*mx, blur(y*y)-my*my, blur(x*y)-mx*my
    ssim = ((2*mx*my+6.5025)*(2*vxy+58.5225))/((mx*mx+my*my+6.5025)*(vx+vy+58.5225))
    return dict(mae_8bit=float(np.mean(np.abs(delta))),
                psnr_db=10*math.log10(255**2/max(mse, 1e-12)),
                ssim=float(np.mean(ssim[5:-5,5:-5])))


def compare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dest = output/'episode'
    dest.mkdir()
    metadata = json.loads((source/'manifest.json').read_text())
    if metadata.get('status') != 'kept':
        raise ValueError('Use a completed kept source episode')
    options = dict(format='collector_h265', qp=30)
    resolved = probe(options)
    print('Hardware probe passed:', resolved, flush=True)
    source_bytes = sum(f.stat().st_size for f in source.rglob('*') if f.is_file() and 'replay_tracking' not in f.parts)
    metadata.update(status='converting', schema=FORMAT, observation_storage='h265_explicit_frame_refs',
                    storage_format=FORMAT, compact_records='hil/ticks.csv', converted_from=str(source))
    (dest/'metadata.json').write_text(json.dumps(metadata, indent=2))
    writer = CollectorWriter(dest, metadata['fps'], options)
    timings, samples = [], []
    start = time.perf_counter()
    rows = ticks = 0
    try:
        for row in records(source):
            t = time.perf_counter()
            writer.add(row)
            timings.append(time.perf_counter()-t)
            if row['kind']=='tick':
                ticks += 1
                if row['tick'] % 120 == 0:
                    samples.append((row['tick'], {k:v for k,v in row.get('observation',{}).items()
                                    if is_image(v)}))
            rows += 1
            if rows % 200 == 0:
                print('converted records', rows, 'elapsed', round(time.perf_counter()-start,2), flush=True)
    finally:
        close_start = time.perf_counter()
        writer.close()
    conversion_s = time.perf_counter()-start
    close_s = time.perf_counter()-close_start
    metadata.update(status='kept', video_encoding=dict(codec='hevc_vaapi', qp=30, pixel_format='yuv420p'),
                    image_storage_lossy=True)
    (dest/'metadata.json').write_text(json.dumps(metadata, indent=2))
    # Every scalar, command, event, image identity and reference position must match.
    for i, pair in enumerate(itertools.zip_longest(records(source), iter_records(dest))):
        old, new = pair
        if skeleton(old) != skeleton(new):
            raise AssertionError(f'Event/observation association changed at record {i}')
    print('All records and image associations verified:', rows, flush=True)
    counts = {}
    for stream in writer.archive.streams.values():
        cap = cv2.VideoCapture(str(stream.directory/'video.mp4'))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:break
            count += 1
        cap.release()
        if count != stream.frames:
            raise AssertionError(f'Frame loss in {stream.directory}: {count}/{stream.frames}')
        counts[stream.directory.name] = count
    wanted = dict(samples)
    metrics = {}
    decoder = VideoDecoder(dest)
    preview_saved = set()
    try:
        for row in iter_records(dest):
            if row.get('kind') != 'tick' or row['tick'] not in wanted:continue
            for name, png in wanted[row['tick']].items():
                a = cv2.imdecode(np.frombuffer(png,np.uint8),cv2.IMREAD_COLOR)
                data = decoder.decode(row['observation'][name])
                b = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                metrics.setdefault(name, []).append(dict(tick=row['tick'], **quality(a,b)))
                if name not in preview_saved and row['tick'] >= 120:
                    diff = np.clip(np.abs(a.astype(float)-b)*5,0,255).astype(np.uint8)
                    panels=[]
                    for frame,label in [(a,'Original image'),(b,'H.265 QP30'),(diff,'Absolute difference x5')]:
                        panel=cv2.copyMakeBorder(frame,30,0,0,0,cv2.BORDER_CONSTANT)
                        cv2.putText(panel,label,(10,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                        panels.append(panel)
                    cv2.imwrite(str(output/(name.replace('/','_')+'_comparison.jpg')),np.hstack(panels))
                    preview_saved.add(name)
    finally:decoder.close()
    size = sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
    result = dict(source=str(source), converted=str(dest), source_bytes=source_bytes, new_bytes=size,
                  size_reduction_factor=source_bytes/size, conversion_s=conversion_s, finalization_s=close_s,
                  control_ticks=ticks, records=rows, decoded_video_frame_counts=counts,
                  writer_add_ms={'mean':float(np.mean(timings)*1000),'p99':float(np.percentile(timings,99)*1000)},
                  quality_samples=metrics, quality_summary={k:{metric:float(np.mean([s[metric] for s in v]))
                  for metric in ('mae_8bit','psnr_db','ssim')} for k,v in metrics.items()},
                  associations_exact=True, all_video_frames_decoded=True)
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='quality_samples'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('output')
    args=parser.parse_args()
    compare(args.source,args.output)
