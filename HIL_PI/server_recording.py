"""Retryable commit/discard over the existing ZMQ LAN connection; no robot imports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from HIL_PI.recording import atomic_json, compact_records

UPLOAD_FIELDS = {'tick','mode','segment','state','human_action','policy_action','policy_raw_action',
                 'policy_source','pair_valid','publish_start','publish_time','observation_time',
                 'robot_send_start','robot_send_end','published_action'}


def sync_decision(path, worker):
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['status'] not in ('kept','discarded'):
        raise ValueError('No explicit keep/discard decision; pending cannot be trained')
    txn=json.loads((path/'server_transaction.json').read_text())
    identity={k:txn[k] for k in ('episode_id','token')}
    # End is idempotent, including after disconnect or server restart.
    keep=manifest['status']=='kept'
    batches=0
    try:
        worker.control(dict(cmd='hil_end',**identity))
        if keep:
            batch=[]
            for row in compact_records(path):
                if row['kind']!='tick': continue
                batch.append({k:row.get(k) for k in sorted(UPLOAD_FIELDS)})
                if len(batch)==128:
                    worker.control(dict(cmd='hil_upload',**identity,index=batches,rows=batch))
                    batches+=1; batch=[]
            if batch:
                worker.control(dict(cmd='hil_upload',**identity,index=batches,rows=batch))
                batches+=1
        if keep and manifest.get('config', {}).get('hil', {}).get('expo_online', False):
            from HIL_PI.expo_recording import upload_observations
            upload_observations(path, worker, identity, manifest)
        response=worker.control(dict(cmd='hil_decide',**identity,keep=keep,batches=batches,
            outcome=manifest.get('outcome'),last_tick=manifest.get('terminal_last_tick')))
        txn.update(sync_status='committed',result=response)
        atomic_json(path/'server_transaction.json',txn)
        return response
    except Exception as exc:
        txn.update(sync_status='retry_required',error=str(exc))
        atomic_json(path/'server_transaction.json',txn)
        raise


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('episode',help='A locally kept/discarded episode directory')
    args=ap.parse_args()
    from HIL_PI.control import PolicyWorker
    path=Path(args.episode)
    txn=json.loads((path/'server_transaction.json').read_text())
    worker=PolicyWorker(txn['server'],timeout_ms=120000)
    try:
        print(sync_decision(path,worker))
    finally:
        worker.close()


if __name__=='__main__': main()
