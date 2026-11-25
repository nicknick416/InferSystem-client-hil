"""Explicit success/failure labels for every episode, including human takeover."""
from HIL_PI.recording import compact_records,atomic_json,metadata_path


def choose_episode_decision(episode,keyboard):
    episode.seal()
    last=-1;declared=None
    for row in compact_records(episode.path):
        if row['kind']=='tick':
            last=max(last,int(row['tick']))
        elif row['kind']=='episode_end':declared=row.get('last_tick')
    if last<0 or declared!=last:
        choice=keyboard.wait('nq','本轮没有完整执行记录：n 删除本轮 / q 删除本轮并退出')
        return None if choice=='q' else False
    choice=keyboard.wait('sfnq','请标记本轮任务结果：s 成功并保存 / f 失败并保存 / n 删除本轮 / q 删除本轮并退出')
    if choice=='q':return None
    if choice=='n':return False
    episode.metadata.update(outcome='success' if choice=='s' else 'failure',terminal_last_tick=last,
                            outcome_source='operator_terminal_confirmation')
    atomic_json(metadata_path(episode.path),episode.metadata)
    return True
