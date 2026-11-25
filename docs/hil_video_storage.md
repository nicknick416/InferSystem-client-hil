# HIL 视频采集格式（2026-11-18）

UR 现场 RTC / 非 RTC 配置，以及三个 UR 示例配置，已启用 `hil.recording.format: collector_h265`。
沿用旧采集系统 `/home/xinzhi/code/auto` 的 VAAPI H.265 编码方式，RGB 和触觉均使用 QP 30、
YUV420P、原始分辨率，每秒一个关键帧。实时模型输入不改变；这里改变的是本地存储。
软件编码不静默回退：硬件编码不可用时，在打开机械臂、相机、HID 之前拒绝启动。

```yaml
hil:
  recording:
    format: collector_h265
    qp: 30
    vaapi_device: /dev/dri/renderD128
    # ffmpeg: /path/to/ffmpeg   # 可选；默认搜索 PATH、当前环境及相邻 dc 环境
```

默认读取程序使用的 `infer-client` 环境已有 OpenCV；编码器复用开发机 `dc` 环境的 FFmpeg。
未配置 `hil.recording` 的其他设备配置继续使用原录制格式，旧记录保持可读。
正在运行的程序不会热更新；下一次启动使用新配置。

## 一条新记录

```text
episode目录/
├── metadata.json                       # 成功/失败、配置、schema、编码与完整性信息
├── observation.image.<标识>/
│   ├── video.mp4                       # RGB 或触觉；目录由 wire key 稳定生成
│   ├── timestamps.csv                  # 视频帧号、首次引用 tick、传感器时间、原图哈希
│   ├── config.csv                      # 实际尺寸、帧数、编码器
│   └── encoder.log                     # 编码错误诊断
├── baseline.<相机标识>/                 # 触觉基准帧也以 H.265 保存，仅保存一次
├── history.<相机标识>/                  # 可选：首次出现但未保存过的历史帧
├── actions.eef_pose/data.csv            # tick、发布时间、xyz、qw/qx/qy/qz
├── actions.gripper/data.csv             # tick、发布时间、夹爪宽度（米）
├── observation.state.eef_pose/data.csv
├── observation.state.joint_position/data.csv
├── observation.state.gripper/data.csv
└── hil/
    ├── ticks.csv                       # HIL 完整执行信息；复杂字段是 JSON 单元格
    ├── observations.jsonl              # 每个 tick 的观测结构与显式视频帧引用
    ├── events.jsonl                    # 完整事件顺序，tick 行索引及预测/切换等事件
    ├── pairs.jsonl                     # 同 tick 人工动作与影子策略配对
    └── execution_segments.jsonl         # 按策略来源划分的执行片段
```

新格式不再生成 `control.msgpack`、`records.msgpack`、`pairs.msgpack`、
`execution_segments.msgpack`、`blobs/data.bin` 或 `manifest.json`。
`schema` / `storage_format` 为 `collector_h265_v1`。这是借鉴旧采集系统的 HIL 扩展格式，
不是声明与旧采集系统所有上传/训练工具无修改兼容；尤其四元数列明确为 wxyz。

## HIL 对齐约定

- 动作、关节角、夹爪、状态、控制时刻、推理请求来源、接管模式和配对有效性仍按原值记录。
- `hil/ticks.csv` 保留完整语义；各 modality CSV 提供便于直接分析的数值视图。
- RGB 和触觉当前图像各按控制 tick 写一帧，重复观测可以重复编码；帧间编码会压缩这些重复。
- 历史帧以明确的视频文件和帧号引用，复用已保存图像；未见过的历史帧单独保存。
  基准帧单独保存。即使图像相同，也不会把“当前帧”“历史帧”“基准帧”的关联结构混为一谈。
- 视频是固定名义帧率，精确时间必须读取原始 `publish_start`、`state_time`、`sensor_times`
  与引用，不能简单把某路视频第 N 帧当作任意事件的第 N 条观测。
- 原图 SHA-256 用于追踪来源，不是压缩后像素的校验和。解码图像有损，不可能恢复原编码字节。

`HIL.recording.records()` / `compact_records()` 同时支持新旧格式；HIL_PI 同步适配。
默认 `records()` 会把视频帧解码回 PNG 字节，恢复原 payload 结构；
`records(resolve_blobs=False)` 返回帧引用，不解码图像。
`compact_records()` 不解码视频，供 replay、配对整理、结束检查及服务器动作同步使用。
EXPO 观测上传也通过新读取接口解码。

## 保存、失败和完整性

编码在录制后台线程与各路 FFmpeg 子进程中进行，控制线程只向有界队列提交记录。
保留容量 128 的队列，满队列报错，不静默丢弃动作或图像。编码进程退出、写管道超时、
收尾失败或磁盘同步失败均拒绝把本轮标为已完成。失败记录保留为 `.pending_...`。
编码器收尾等待上限 30 s；管道写入超时 10 s。

MP4 封装完成后执行文件同步，CSV/JSONL 也同步落盘，再显示保存选择。
没有通过删除 `fsync` 来伪装保存完成。`metadata.json` 中记录队列峰值、视频帧数和收尾耗时。

## Replay

原入口和参数保持可用，输入新 episode 目录即可；不需要读取视频来重放机械臂动作。

```bash
cd /home/xinzhi/InferSystem-client-hil
conda activate infer-client
python Example/ur/replay_episode.py Config/local/ur_loading_rtc_hil.yaml /path/to/episode --dry-run
# 现场确认后，再加 --live；它会自动移动到片段起点。
```

## 同一条原始数据的验证

原记录：`data/20261117_173611_211177/20261117_173804_827098_episode_000004`，
714 个控制 tick，785 条总事件，约 23.77 秒。原始数据未修改。

开发机的离线比较结果在 `validation/h265_20261118_v2/`：

- 原数据 701,156,734 字节；转换数据 14,571,552 字节，约 48.1 倍体积差。
- 转换全过程（含读取旧数据）约 10.95 秒，编码/落盘收尾约 0.58 秒。
- 四路视频各 714 帧，两路基准各 1 帧；逐帧解码验证完整。
- 所有事件、非图像字段、原图身份及观测引用位置逐项核对一致。
- 每路每 120 tick 抽样一次，共 6 帧，计算压缩图与原图解码结果的差异：

| 视图 | 平均绝对误差（0–255） | PSNR（dB） | SSIM |
|---|---:|---:|---:|
| 外部 RGB | 2.66 | 36.68 | 0.951 |
| 腕部 RGB | 2.94 | 36.01 | 0.931 |
| 左触觉 | 4.65 | 32.34 | 0.889 |
| 右触觉 | 4.62 | 32.52 | 0.877 |

触觉图像的细纹理变化比 RGB 更明显。此指标是抽样图像相似性，不代表下游训练性能验证。
原始触觉输入已为 JPEG，RGB 为 PNG；均解码后进入 H.265。

开发机在 `validation/h265_realtime_20261118/` 保存了同一记录以 30 Hz 定时注入的后台采集测试：
队列峰值 12，前台提交平均 0.039 ms、最大 0.133 ms，结束 seal 约 0.80 s，
保存整理约 0.12 s。该完整新回合约 14.89 MB（包括派生配对/片段文件）。
command / measured 两种 replay 模式的目标、时间与夹爪数组均与旧记录完全一致。
此测试没有机械臂运动，也没有相机和模型并行负载；现场整机负载下的收尾时间仍需验证。

离线转换工具：`Example/ur/compare_storage.py SOURCE_EPISODE NEW_OUTPUT_DIRECTORY`。
输出目录必须不存在，工具不会覆盖或删除原始记录。

编码器实测测试需显式启用（不访问机器人）：

```bash
HIL_TEST_VAAPI=1 python -m pytest -q tests/test_video_storage.py
```

## 上下料 SpaceMouse 人控的静止段

本地 UR VTLA 上下料 RTC / 非 RTC 配置启用 `hil.omit_settled_human_frames: true`。设为 `false` 可恢复完整逐帧录制。在同一次人控中操作过手柄或按键之后，松手且连续稳定至少 0.30 秒，写入器才省略后续重复帧。稳定判断同时比较实测末端位置（0.5 mm）、旋转与各关节（0.15°）、实测夹爪（0.5 mm）及发布目标是否保持不变，并使用固定参考而非仅比较相邻帧。保留起始过渡以及恢复操作、交回策略或回合结束前的最后一帧。中途长暂停同样仅省略稳定的内部，按键、推杆、策略帧和结束结果照常保存。过滤位于本地写入线程、视频编码之前；仅适用于本地 SpaceMouse H.265 录制，不用于键盘 HIL、服务端采集或 EXPO。

`tick` 和原始时间戳始终指物理控制时刻，因此 tick 可能不连续；`frame_index` 是从零开始的连续已保存行号，与逐行观测和当前视角视频对齐。所有事件及影子推理仍按原控制时刻记录。`metadata.json` 中 `total_frames` 为保存行数，`source_tick_count` 为执行总帧数，`omitted_human_idle_frames`、`omitted_human_idle_ranges` 可追溯省略的数量和原始 tick 区间（含两端）；`terminal_last_tick` 是结束时的真实控制 tick。训练动作块不能跨原始 tick 缺口或控制模式切换，不能把相邻的两行直接当作相邻控制步。回放工具保留原有缺帧保护，使用连续区间回放；要完整逐时刻回放则关闭本开关录制。以前录制的数据格式不变。

### UR 上下料：保存连续训练轨迹

四个 UR 配置现在启用 `hil.omit_stationary_frames` 和 `hil.export_dagger_training`。静止检测在采集写盘线程进行，覆盖开头、中途和末尾的策略与人控，不改变推理、机械臂指令或实际控制频率。每段确认静止保留第一帧；短暂停顿须指令几乎完全不变、实测只有微小抖动。较宽的噪声容差须稳定至少 0.30 秒，位置、姿态、各关节、实测夹爪宽度和下发目标都以固定起点比较。输入设备仍在操作、夹爪仍有开合、目标尚未跟上、能分辨的缓慢定向运动或反馈过期时保留。切换控制者或接管段会重置检测。完全静止的回合仍有一个真实观测，而非空 episode。

`source_tick_count` 和 `source_terminal_last_tick` 保留实际执行长度和最后控制 tick；`terminal_last_tick` 是最后保存的 tick。`omitted_stationary_ranges` 记录每段被省略的原始 tick、模式及时间，`omitted_policy_idle_frames` / `omitted_human_idle_frames` 为各自数量。导出器只跨越由写盘器核实的静止缺口，保留恢复运动的第一帧。旧配置及旧回合仍使用各自原有规则。

选择保留的成功或失败回合，在 `data/<本次采集时间>/training/` 下自动生成两个分别可用的 LeRobot v2.1 数据集：

- `full_episode/`：默认训练入口。每帧动作都是当时实际下发的 32 维指令；策略、人控、再回策略都留在同一条时间线上。切换控制权不会切段，真实的动作变化不会被平滑。来源 tick 缺失且不能证明只是静止、或反馈时间异常时，另起 episode。
- `human_only/`：同一次接管的连续动作组成一个 episode；中间策略控制过或是下一次人工接管，就另起 episode。无接管的回合不会出现在这里。两套数据有重叠，按训练目的选用其中一套。

四路视频、状态和动作按相同帧索引保存。每段训练时间从零开始按 30 Hz 排列，`source_tick` 可回溯原始时刻，`is_human` 只标注来源。人控接管自动重置夹爪目标时，在 `human_only` 中保留实际动作并暂时关闭该维的 `action_mask`，直到操作者按键或退出接管。原始回合仍保留真实时钟，元数据记录静止裁剪范围与数量。每条训练轨迹在 `meta/episodes.jsonl` 的 `outcome` 保留原始 `success` 或 `failure` 标签；二者使用相同的连续性和静止过滤规则。丢弃的回合不会进入训练集。

训练读取器无需改动。若既有训练配置以每个 episode 的首帧作为触觉参考，`human_only` 的参考就是本次接管开始时的触觉图，可能已经接触物体；现场采集的双侧触觉基准另存于 `baselines/`，现有读取器不会自动使用它。复制整个数据集时保留 `.versions/` 与 `meta` 链接，或在复制时跟随符号链接。导出中断不会丢弃保留的回合；在采集机项目目录运行 `python -m HIL.dagger_export <回合目录>` 可补导，已成功的回合不会重复加入。

采集机当前的 `infer-client` 环境已包含自动导出所需的 `av`、`pyarrow` 和 `Pillow`。新建环境如需开启这两项保存选项，须安装相应依赖并配置采集机可用的 H.265 编码器。
