"""Upload real per-tick observations at fixed-C replay boundaries after robot stop."""
from pathlib import Path
from HIL.recording import compact_records, records, decode_observation
from HIL.blobs import decode


def upload_observations(path, worker, identity, manifest):
    path = Path(path)
    c = int(manifest['config']['hil']['expo_replan_steps'])
    if c < 1:
        raise ValueError('EXPO replan steps must be positive')
    remote = worker.control(dict(cmd='expo_info'))
    if remote['contract']['config']['replan_steps'] != c:
        raise ValueError('Client/server EXPO chunk lengths differ')
    ticks = {int(r['tick']) for r in compact_records(path) if r['kind'] == 'tick'}
    last = int(manifest['terminal_last_tick'])
    if ticks != set(range(last + 1)):
        raise ValueError('Incomplete recorded execution sequence')
    wanted = set(range(0, last + 1, c))
    if last + 1 >= c:
        wanted.add(last + 1 - c)
    for row in records(path, resolve_blobs=False):
        if row['kind'] != 'tick' or int(row['tick']) not in wanted:
            continue
        tick = int(row['tick'])
        observation = decode_observation(row['observation'], path)
        worker.control(dict(cmd='expo_observe_upload', **identity, tick=tick, observation=observation))
        wanted.remove(tick)
    if wanted:
        raise ValueError(f'Missing recorded observations: {sorted(wanted)}')
