# UR HIL 适配

VTLA 入口 `robot_inference_hil.py` 支持单臂 UR + Robotiq、32D canonical 动作，提供 RTC 与非 RTC 配置。`robot_inference_pi_hil.py` 同样支持 UR，但保留独立的 PI 10D RTC 协议；必须连接匹配的 PI 服务和 checkpoint。Flexiv 专用 DSRL cord collector 不用于 UR。

## 开发机运行

在 `xinzhi-dev` 上：

```bash
cd /home/xinzhi/InferSystem-client-hil
conda activate infer-client
python robot_inference_hil.py Config/local/ur_loading_rtc_hil.yaml --check-only
python robot_inference_hil.py Config/local/ur_loading_nonrtc_hil.yaml --check-only
```

现场配置复制自 `/home/xinzhi/InferSystem-client/Config/local/ur_loading_{rtc,nonrtc}.yaml`，沿用 UR 地址、两路 RealSense、两路触觉、Home、任务提示词和推理端点。它们位于忽略目录 `Config/local/`，公开示例为 `Config/ur_vtla_rtc_hil.yaml`、`Config/ur_vtla_nonrtc_hil.yaml`。原参考项目未修改。

确认设备、服务和作业空间后，选择其中一种启动：

```bash
python robot_inference_hil.py Config/local/ur_loading_rtc_hil.yaml --live
# 或非 RTC：
python robot_inference_hil.py Config/local/ur_loading_nonrtc_hil.yaml --live
```

不带 `--live` 默认只检查配置和依赖，不打开机器人、相机、HID 或服务连接。`--live` 会连接硬件并激活夹爪，随后 Enter 确认回起始位姿，再按 Enter 开始一轮。任务提示默认读取 YAML 的 `inference.prompt`，可用 `--prompt` 覆盖。

移动 SpaceMouse 接管；左键闭合、右键打开，松手保持，`p` 交回策略。沿用原 HIL 的基坐标系平移映射；UR 示例关闭旋转轴，轴向需现场核对后再启用。`r` 结束本轮并回位，然后确认结果；控制中的 `q` 停止当前回合，确认保存或丢弃后退出，不自动回 Home 或打开夹爪。Ctrl+C 异常停止。数据写到项目下的 `data/启动时间/回合时间_episode_序号/`，具体按键见文末的采集保存说明。

## 控制与数据约定

- UR 六关节，TCP 为 xyz 米 + wxyz 四元数；canonical 状态/动作前十维为 xyz、旋转矩阵前两列、夹爪，余下补零。
- `normalized_width` 的状态夹爪仍为米，动作夹爪为宽度 / `max_width`，0 闭合、1 全开。人工接管、等待保持、回退、候选预览、实际发布记录均使用同一换算，RTC 前缀保持模型动作单位。
- 禁止把 `normalized_width` 与十倍米转换同时启用。Robotiq 使用项目内驱动，默认不自动开合标定，`set(True)` 为打开。
- UR `control.frequency_hz` 与策略 `fps` 都为 30，`wait_step_reached: false`。配置检查拒绝频率不一致、非六维 Home、禁用夹爪等情况。
- Home 使用 UR `moveJ`，恢复停止后的 RTDE 脚本，检查调用返回值、实际关节到位及静止状态；超时停止并拒绝开始推理。`hil.home_velocity_scale` 在 UR 中传给 `moveJ` 的关节速度，单位 rad/s，现场配置为 0.3。
- `servoL` 返回失败时抛出异常，避免把拒绝执行的目标记录为成功下发。HIL 保留原发布时刻记录。

## RTC 与参考客户端的差异

HIL 调度使用 `inference.rtc.enabled/prefix_length`，不是普通客户端的 `rtc_prefix`。现场 RTC 为最多 8 步，首次空队列发送 0 步；非 RTC 显式关闭。

HIL 的 RTC 协议要求前缀保持一致，配置校验会拒绝正的 `max_eef_xyz_step`、`max_eef_rot_step_deg`、`max_eef_gripper_step` 以及策略运动裁剪。因此 RTC 配置将这三项设为 0，并保留实际状态到目标的位置/姿态拒绝阈值（继承参考配置的 0.3 m / 1 rad）。它不等同于参考客户端每步 2° 的旋转裁剪。需要更严格限制时应调整拒绝阈值；非 RTC 保留参考配置的逐步裁剪参数。以上数值继承自现场配置，并未通过本次实机验证。

图像仍由 HIL 的 CameraWorker 与 TactileRingBuffer 采集；不迁移普通客户端的 `background_observation` 开关。HIL 保留自身队列、接管仲裁、录制和 RTC 响应校验，不替换为普通客户端主循环。

## 检测与可选功能

现场配置开启状态反馈停滞监测，默认关闭自动回退、触觉异常检测和外部修正。停滞阈值来自现有 HIL 示例，需要结合 UR 记录校准；触觉图像仍发送给主模型。未复制 Flexiv 的正常触觉样本、修正服务或令牌。

要使用回退，在本地配置启用 `hil.stuck_recovery.enabled`，核对距离、速度和历史参数；这会同时允许检测触发的自动回退，不能把它视为纯手动开关。`b` 还要求 `manual_recovery_enabled: true`。键盘双按流程可用既有 `hil.keyboard_hil` 配置，需按原校验关闭自动检测/回退/修正。

`Config/ur_pi_rtc_hil.yaml` 是脱敏的 PI 模板，检查命令：

```bash
python robot_inference_pi_hil.py Config/ur_pi_rtc_hil.yaml --check-only
```

该模板未绑定现场 PI 服务；不能直接当作当前 VTLA 服务的启动配置。

## 离线验证（2026-11-10）

修改前全套测试：353 通过、13 失败；修改后：378 通过、13 失败，失败项目一致。原有失败涉及 Aloha、旧 DSRL 协议测试和缺失的 `Example.realsense_visualize`，本次没有修复这些独立问题。

新增验证覆盖两套入口的 UR 配置/依赖选择、夹爪单位、人工动作与候选预览、实际下发/RTC 前缀、UR 回 Home 成功/拒绝/超时、servoL 拒绝及完整 HIL 主循环中的回退/接管/恢复。两个现场 VTLA 配置与 PI 模板的 `--check-only` 均通过。

本次没有连接或移动真实机器人，没有验证相机实际可用性、推理服务协议、SpaceMouse 轴向或真实任务成功率。


## 2026-11-16 SpaceMouse 人工平移速度

根据现场手感反馈，RTC / 非 RTC 两份 UR 配置恢复原 measured 平移方式：
每个控制周期从当前实测 TCP 位置生成增量目标，不累加上一周期的平移目标。
保留提高后的 linear_speed_m_s: 0.21（原值 0.06），控制频率仍为 30 Hz；
单轴满量程对应当前实测位置加 7 mm，不代表实际末端必然达到 0.21 m/s。
translation_mode: measured 已在两份现场配置中显式设置。

松手保持、旋转开关、夹爪速度和数据记录流程沿用原逻辑。
integrated 模式代码保留用于兼容和重放历史记录，当前现场配置不启用。
正在运行的进程不会热更新。按 q 正常退出后，使用原启动命令重启生效。

## 采集保存目录（2026-11-17）

所有随项目提供的 HIL 配置默认使用 `hil.output_dir: data`，相对路径按项目根目录解析。每次 `--live` 启动建立一个北京时间目录，例如：

```text
/home/xinzhi/InferSystem-client-hil/data/
└── 20261117_153000_123456/
    ├── 20261117_153015_234567_episode_000001/
    └── 20261117_153100_345678_episode_000003/
```

回合序号从 1 开始，丢弃的尝试也占用序号；重新启动会创建新目录并从 1 开始。成功和失败使用相同命名规则，`manifest.json` 的 `outcome` 分别记录 `success` 或 `failure`，并保留 `episode_index`、`session_name` 和带时区的开始时间。

无论是否发生人工接管，回合结束后统一选择：`s` 成功并保存，`f` 失败并保存，`n` 删除整个回合，`q` 删除当前回合并退出。成功和失败都保留原有图像、控制记录、配对记录和轨迹来源。正在写入的回合暂用隐藏的 `.pending_...` 目录，确认结果后改为正式名称；异常中断留下的临时回合不能当作已完成数据使用。

丢弃不再留下 manifest、图片、配对或服务器事务文件。若启用 server capture，删除前发送服务器丢弃决定；通信失败会报错，但本地已丢弃数据仍删除。成功/失败回合提交服务器失败时，仍保留本地数据和可重试事务。

`--check-only` 不创建采集目录。此前已保存的数据不迁移、不删除。
