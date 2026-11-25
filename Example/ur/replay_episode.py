"""Replay a recorded UR HIL episode; defaults to an offline check."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HIL.ur_replay import load_episode, replay


def main(argv=None):
    parser = argparse.ArgumentParser(description='UR HIL episode 回放（默认离线检查）')
    parser.add_argument('config', help='UR YAML 配置')
    parser.add_argument('episode', help='episode 目录（支持 H.265/CSV 新格式和旧格式）')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live', action='store_true', help='连接 UR，移动到起点并执行回放')
    mode.add_argument('--dry-run', action='store_true', help='仅读取和检查数据（默认）')
    parser.add_argument('--source', choices=['command', 'measured'], default='command',
                        help='command: 已下发目标；measured: 实测 TCP/夹爪轨迹')
    parser.add_argument('--speed', type=float, default=1., help='时间倍率，0.5 为半速')
    parser.add_argument('--start', type=int, default=0, help='起始 tick 行索引，包含')
    parser.add_argument('--end', type=int, help='结束 tick 行索引，不包含')
    parser.add_argument('--no-move-to-start', action='store_true', help='跳过 MoveJ，要求已在起点')
    parser.add_argument('--start-velocity', type=float, default=.15, help='起点 MoveJ 速度 rad/s')
    parser.add_argument('--max-position-error', type=float, default=.05, help='最大跟踪位置误差 m')
    parser.add_argument('--max-rotation-error', type=float, default=.35, help='最大跟踪姿态误差 rad')
    parser.add_argument('--max-gap', type=float, default=.25, help='允许的最大记录间隔 s')
    parser.add_argument('--max-lag', type=float, default=.25, help='最大控制调度延迟 s')
    parser.add_argument('--settle-timeout', type=float, default=3., help='终点到位等待上限 s')
    parser.add_argument('--log-dir', help='新的输出目录，必须尚不存在')
    args = parser.parse_args(argv)
    try:
        trajectory = load_episode(args.episode, source=args.source, start=args.start, end=args.end,
                                  max_gap_s=args.max_gap)
        replay(args.config, trajectory, speed=args.speed, live=args.live, log_dir=args.log_dir,
               move_to_start=not args.no_move_to_start, start_velocity=args.start_velocity,
               max_position_error=args.max_position_error, max_rotation_error=args.max_rotation_error,
               max_lag_s=args.max_lag, settle_timeout=args.settle_timeout)
    except KeyboardInterrupt:
        print('回放已中断。', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'回放失败：{exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
