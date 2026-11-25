# UR episode 回放

从 `data/启动时间/回合时间_episode_序号/` 读取 UR HIL / HIL_PI 记录，
沿用 ARX5 / Flexiv 示例的「加载 → 起点定位 → 定时回放 → 跟踪日志」流程。
本入口支持新格式 `metadata.json` + `hil/ticks.csv`，以及旧格式 `manifest.json` + `control.msgpack`。
不读取图像，也不启动模型、相机或 SpaceMouse。新保存格式详见 `docs/hil_video_storage.md`。
它不是 Parquet 关节轨迹回放器，不能直接输入 ARX5 / Flexiv 的 Parquet 文件。

## 开发机上运行

```bash
cd /home/xinzhi/InferSystem-client-hil
conda activate infer-client

# 已保存的一条记录；也可以替换成其他 episode 目录
EPISODE=data/20261117_173611_211177/20261117_173804_827098_episode_000004

# 默认只做离线检查；等价于加 --dry-run，不打开任何硬件
python Example/ur/replay_episode.py Config/local/ur_loading_rtc_hil.yaml "$EPISODE"

# 半速回放实际下发命令；--live 会连接硬件并自动移动到起点
python Example/ur/replay_episode.py Config/local/ur_loading_rtc_hil.yaml "$EPISODE" --speed 0.5 --live

# 仅回放第 100 到第 299 行 tick；先离线检查，执行时再加 --live
python Example/ur/replay_episode.py Config/local/ur_loading_rtc_hil.yaml "$EPISODE" --start 100 --end 300 --speed 0.5

# 按当时实际测到的末端/夹爪轨迹回放；先离线检查
python Example/ur/replay_episode.py Config/local/ur_loading_rtc_hil.yaml "$EPISODE" --source measured --speed 0.5
```

执行 `--live` 前先退出占用同一 UR 的推理或其他控制程序，确认起点移动路径可通行，
并保持示教器 TCP、工具设置和机器人基座与录制时一致。起点 `moveJ` 不是沿记录轨迹倒放，
而是从当前关节位置移动到所选片段首帧的实测关节位置，默认速度 0.15 rad/s。
`--no-move-to-start` 跳过这次移动，但要求各关节已经在起点 0.05 rad 范围内，且 RTDE 控制脚本已运行。

## 回放语义

- 默认 `--source command`：读取 `published_pose`（xyz 米 + wxyz 四元数）和
  `commanded_gripper_m`，按 `publish_start` 的相对时间播放，包含策略、人工接管和保持等 tick。
  不播放未执行的模型候选，也不重新经过推理平滑/裁剪流程。
- `--source measured`：读取 `robot_state.eef_pose`、`state[9]`（实测夹爪米）和 `state_time`。
  这会把原来的实测轨迹作为新的控制目标，与重新发送原命令的含义不同。
- `--start` 包含，`--end` 不包含；二者是控制 tick 行索引，非秒数，不包含其他事件行。
- `--speed 0.5` 将记录时长拉长到两倍。控制周期由 YAML 的 `robot.control.frequency_hz`
  决定（现场配置 30 Hz），xyz 线性插值、四元数最短路径 SLERP，夹爪保持最近一个历史命令。
  记录时刻与控制栅格不同，因此这是按原时间线重采样，不是逐字节逐帧复制。
- 末端通过现有 UR `servoL` 接口发送，不使用非实时 `moveL` 降级路径。夹爪使用米到
  Robotiq 原始位置的现有换算，并检查命令接受结果。
- 不使用 `pairs.msgpack`：它只覆盖部分人工纠正配对，不能代表整段轨迹。
- 正常结束、Ctrl+C 或执行异常都会尝试停止并断开；不自动清除故障、重试、不回 Home，
  不额外打开夹爪。命令回放不能保证复现相同的接触力、物体状态或任务结果。

## 检查与停止条件

只接受已保存的 `status: kept` UR 回合；拒绝临时回合、缺失 tick、非递增时间、非有限数值、
非法四元数和夹爪宽度。默认记录最大间隔为 0.25 s。
实际配置的夹爪量程/方向需与录制一致，自动夹爪标定须关闭。

默认位置跟踪误差上限为 0.05 m，姿态误差上限为 0.35 rad；也用这两个值检查源轨迹相邻跳变和起点 TCP。
出现故障、通信失败、控制脚本停止、动作拒绝、跟踪超限或调度延迟超过 0.25 s 时中止。
终点持续发送最后一个目标，等待连续三次观测进入 5 mm / 0.05 rad（或更严格的用户阈值）内，
默认最多等待 3 s；这只验证末端到位，不宣称夹爪已经夹持成功。

参数：`--max-position-error`（米）、`--max-rotation-error`（弧度）、`--max-gap`（秒）、
`--max-lag`（秒）、`--settle-timeout`（秒）、`--start-velocity`（rad/s）。
完整列表见 `python Example/ur/replay_episode.py --help`。

## 输出

实机执行默认新建 `episode/replay_tracking/北京时间/`，也可用 `--log-dir` 指定不存在的新目录：

- `tracking.csv`：每次发送前的目标/实测末端位姿、位置/姿态误差、夹爪目标/实测宽度、源 tick 和时间。
  误差反映发送前的跟踪滞后，不是同一命令发送后的瞬时响应。
- `summary.json`：完成/失败/中断状态、发送帧数、误差峰值、失败原因和清理错误。

输入的原始记录不会被修改。离线检查不新建回放输出目录。

## 代码与离线测试

入口：`Example/ur/replay_episode.py`；可复用逻辑：`HIL/ur_replay.py`。

```bash
python -m pytest -q tests/test_ur_replay.py tests/test_ur_robot.py tests/test_ur_home_script_ready.py tests/test_ur_hil.py
```

测试使用模拟机械臂与时钟，不连接真机。实机轨迹精度和接触任务效果需要现场验证。
