"""Explicit task outcome for pure-policy episodes; never infer failure from reset."""
from HIL.recording import compact_records,atomic_json


def choose_episode_decision(episode,keyboard):
    episode.seal()
    human=False;last=-1;declared=None
    for row in compact_records(episode.path):
        if row['kind']=='tick':
            human |= row['mode']=='human'
            last=max(last,int(row['tick']))
        elif row['kind']=='episode_end':declared=row.get('last_tick')
    expo = episode.metadata.get('config', {}).get('hil', {}).get('expo_online', False)
    if human and not expo:
        choice=keyboard.wait('ynq','保留上一轮 HIL human/policy pair？y 保留 / n 丢弃 / q 保持 pending 并退出')
        return None if choice=='q' else choice=='y'
    if last<0 or declared!=last:
        choice=keyboard.wait('nq','本轮没有完整执行记录：n 丢弃 / q 暂存退出')
        return None if choice=='q' else False
    choice=keyboard.wait('sfnq','请标记本轮任务结果：s 成功并保留 / f 失败并保留 / n 丢弃 / q 暂存退出')
    if choice=='q':return None
    if choice=='n':return False
    episode.metadata.update(outcome='success' if choice=='s' else 'failure',terminal_last_tick=last,
                            outcome_source='operator_terminal_confirmation')
    atomic_json(episode.path/'manifest.json',episode.metadata)
    return True
