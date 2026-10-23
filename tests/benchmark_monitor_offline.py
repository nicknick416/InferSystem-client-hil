"""Replay saved observations to Monitor only; no robot imports or commands."""
import argparse
import json
from pathlib import Path
import time
import msgpack
import numpy as np
from HIL.monitor_client import RPCSession, validate_config
from HIL.recording import records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode')
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--output', default='/tmp/monitor_benchmark.json')
    args = parser.parse_args()
    cfg = validate_config(dict(request_timeout_s=30))
    session = RPCSession(cfg)
    rows = []
    for row in records(args.episode):
        if row.get('kind') != 'tick' or row['tick'] % args.stride:
            continue
        request = dict(cmd='predict_monitor', payload=row['observation'],
                       request_id=row['tick'], generation=0, episode_id='offline')
        start = time.monotonic()
        response = session(request)
        rows.append(dict(tick=row['tick'], elapsed_ms=(time.monotonic()-start)*1000, **response))
        if len(rows) == 1:
            Path('/tmp/monitor_sample.msgpack').write_bytes(msgpack.packb(request, use_bin_type=True))
        if len(rows) >= args.limit:
            break
    successful = [r for r in rows[2:] if r['status'] == 'ok']
    session.close()
    summary = dict(count=len(rows), errors=sum(r['status'] != 'ok' for r in rows),
                   trigger_count=sum(r.get('trigger', False) for r in rows))
    for key in ('elapsed_ms', 'processing_ms'):
        summary[key] = dict(zip(('p50', 'p95', 'max'),
            np.percentile([r[key] for r in successful], [50, 95, 100]).tolist()))
    Path(args.output).write_text(json.dumps(dict(summary=summary, rows=rows), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
