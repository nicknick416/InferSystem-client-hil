# InferSystem HIL Client

面向算法开发的机器人 Human-in-the-Loop 客户端：异步调用主策略，执行动作队列，检测运动停滞与触觉异常，并支持人工接管、历史回退和干预数据记录。本分支 `hil` 包含 VTLA 与 OpenPI 两套 HIL 入口，主策略和可选的腕部修正模型在外部服务运行。

这是一套实验框架。检测输出是算法证据，不是机器人安全认证；离线测试通过不等于真实任务成功率已验证。

## 架构与代码入口

```text
机器人反馈 + RGB/触觉相机
          ↓
观测快照 → 异步推理 worker → 动作队列 / RTC 前缀
          ↓                         ↓
运动停滞检测 + 触觉独立进程 → 控制仲裁 ← SpaceMouse / 键盘
                                    ↓
                         动作检查与 dispatcher → 机器人
                                    ↓
                 控制记录 / 人工与策略配对 / 修正样本
```

| 模块 | 职责 |
|---|---|
| `HIL/runtime.py` | VTLA HIL 主循环、事件处理与组件生命周期 |
| `HIL_PI/runtime.py` | OpenPI HIL 主循环及可选 DSRL 接口 |
| `HIL/control.py`、`HIL/rtc.py` | 单个在途推理请求、模式仲裁、动作对齐和 RTC 前缀 |
| `HIL/stuck_detection.py`、`feedback_stuck_detection.py`、`chunk_stuck_detection.py` | 指令与实测反馈的停滞检测 |
| `HIL/tactile_process.py`、`tactile_stuck_detection.py` | 最新触觉帧传输、独立进程检测与结果时效检查 |
| `HIL/tactile_normal.py`、`tactile_slip.py` | 局部接触形变与光流滑动证据 |
| `HIL/stuck_recovery.py`、`keyboard_hil.py` | 历史位姿回退和键盘干预流程 |
| `HIL/correction.py` | 外部修正服务请求、人工修正样本与可选自动执行 |
| `HIL/recording.py`、`blobs.py` | episode、观测和人工/策略配对记录 |
| `Core/`、`Robot/`、`Sensor/`、`Inference/` | 配置、设备抽象、传感器与动作下发 |

`HIL_PI/` 包含对应模块的独立实现；修改共用算法时应检查两套入口并运行各自测试，不能假定修改 `HIL/` 会自动同步到 OpenPI。DSRL、EXPO 和独立修正服务是可选实验路径，基础 HIL 不要求同时启用。

## 安装与配置检查

建议 Python 3.10+。安装通用依赖，机器人 SDK、相机驱动和 HID 权限按所用设备单独准备：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install pytest hidapi
python -m pytest -q tests
```

所有提交的 YAML 都是脱敏示例。复制到 Git 忽略的 `Config/local/` 后，替换机器人/相机序列号、触觉设备路径、机器人网络接口、推理地址和令牌文件。检查起始关节角、夹爪单位、动作维度及位移限制是否适配自己的设备和模型。示例中的回环地址只用于占位，数值阈值也不是通用标定结果。

```bash
mkdir -p Config/local
cp Config/flexiv_vtla_hil.yaml Config/local/hil.yaml
python robot_inference_hil.py Config/local/hil.yaml --check-only
```

`--check-only` 是 HIL 入口的默认行为，不连接机器人、相机、HID 或推理服务；缺少实时依赖时会报告失败。它只能验证配置和依赖可发现性，不验证设备连接或模型协议。其他普通推理入口的 `--dry-run` 不具备同样的无硬件保证。

完成设备核对后，可显式启动实时模式：

```bash
python robot_inference_hil.py Config/local/hil.yaml --live --prompt "Your task instruction."
# OpenPI 入口：使用对应的 OpenPI YAML 和服务。
python robot_inference_pi_hil.py Config/local/pi_hil.yaml --live --prompt "Your task instruction."
```

`--live` 会打开硬件，交互确认后执行回起始位姿和推理。离线开发与 CI 只运行测试或 `--check-only`。

| 示例配置 | 用途 |
|---|---|
| `flexiv_vtla_hil.yaml` | 普通 VTLA + SpaceMouse HIL |
| `flexiv_vtla_rtc_hil.yaml` | VTLA RTC 模式 |
| `flexiv_vtla_expo_hil.yaml` | EXPO/修正相关集成及回归测试参考 |
| `flexiv_tactile_hil.example.yaml` | feedback 停滞检测、local-normal 形变及光流检测 |
| `flexiv_vtla_keyboard_hil.yaml` | 两次按键的人工介入流程 |
| `flexiv_pi_cylinder_rtc_hil.yaml` | OpenPI RTC HIL |
| `flexiv_pi_cylinder_rtc_keyboard_hil.yaml` | OpenPI RTC + 键盘介入 |

发布示例将 `correction.enabled` 和 `correction.auto_execute` 设为 `false`。启用修正前先创建本地令牌文件，并配置服务地址；local-normal 触觉分支仍需要匹配的修正服务就绪信号。自动回退由独立的 `hil.stuck_recovery.enabled` 控制；启用检测不代表必须启用自动动作。

## 异步推理、RTC 与控制归属

推理 worker 只有一个在途请求，控制循环继续执行已接受的动作；人工接管时主策略可作为影子候选继续计算，但不拥有实际机器人控制权。记录中区分模型原始动作、混合后的候选、实际下发命令和实测状态。

RTC 前缀是在采集观测时，从尚未执行的旧动作队列头部取出的动作承诺，不是上一段预测的固定尾部。响应按观测后实际消耗的策略步数对齐，避免简单用推理耗时乘 FPS 推断执行位置。`n_execute` 控制接收的有效动作范围，并不等同于每隔固定秒数同步请求。普通模式、RTC 模式及 EXPO 选项的调度条件各自独立。

模式切换通过 generation 隔离失效响应。RTC 在人工接管、回退和恢复时清除旧前缀；普通模式的 policy→human 转换可保留影子策略时间线。恢复策略后先保持最近命令，等待新观测和新响应，人工影子动作不会成为 RTC 执行承诺。

## 卡住检测：指令要求运动，但反馈未产生相应位移

检测器比较**上一条已发布目标**与当前实测 TCP 位置，不能用本周期刚生成的动作来解释已经发生的反馈。它还检查命令新鲜度、采样间隔、控制模式和目标方向的一致性，避免把等待模型响应或正常保持误判成卡住。

`hil.stuck_detection.window_mode` 支持：

- `time`：时间窗口内的跟踪误差和实际运动范围，加持续确认时间。
- `executed_chunk`：依据已执行 chunk 的运动和误差证据确认，可要求连续多个 chunk。
- `feedback`：检查连续反馈间隔。示例要求误差至少 1.5 mm、5 个反馈间隔内的位置范围不超过 0.18 mm，并且运动意图方向一致。

feedback 模式中，当前位置首先作为前一条命令的反馈；chunk 边界的反馈归属旧 chunk，随后清空窗口，避免混合两个 chunk。释放阈值与进入阈值可不同，以减少状态抖动。输出包含 `tracking_error_m`、`position_excursion_m`、`samples`、`reason` 和进入/清除事件。

该分支主要检测平移停滞，不覆盖所有旋转卡滞、物体掉落或任务语义失败。误差大但仍在正常运动时不应报警。

## 触觉异常：局部形变与光流滑动检测

触觉运算放在独立进程，使用最新帧共享槽和非阻塞访问，不在控制线程累积图像队列。结果须通过 generation 和时效检查；旧结果不能触发回退。进程不可用时触觉分支报告不可用，状态-动作分支仍可工作。

### 局部正常接触模型

`method: local_normal` 用 episode 开始时的无接触图像作为零参考。人工稳定握持后按 `p` 恢复时采集正常接触示例，使用近期最多 5 个局部分数向量的中位数。图像差分缩小到 80×45，裁剪为 56×35 ROI，再分为 5×8 个块，每块计算平均绝对像素差。

每块正常中心为样本中位数，尺度下限为 `energy_min_hold_score`；上界取以下两项的较大值：正常样本最大值加正向余量、正常中心加 `6 × 1.4826 × MAD`。多个块连续超过上界才形成局部形变报警，避免全局平均掩盖局部变化。具体公式见 `HIL/tactile_normal.py`。

冷启动同时要求正常示例数达到 `normal_min_samples`，以及外部修正服务 `/status` 返回 `ready=true` 且 `data_dir` 与配置匹配。状态不可达或目录不匹配时不报警；这是当前实现的门控条件，不是光流算法的理论要求。模型就绪后停止扩充正常示例，每个 episode 仍重建零参考。正常样本目录独立于 episode，丢弃 episode 不会删除它。

其他可选方法为 `energy`（全局接触能量）和 `change`（时序变化）；不能将这些像素指标解释为标定后的物理力。

### 光流滑动证据

光流使用**原始当前触觉图像**缩放、裁剪后的灰度 ROI，而不是腕部 RGB，也不是机器人 TCP 位移：

1. 在上一帧检测最多 80 个角点，使用稀疏金字塔 Lucas–Kanade 跟踪。
2. 反向跟踪检验，剔除前后误差超过 1 像素、越界或无效的点。
3. 以真实帧时间差将位移转成 ROI 像素/秒；从足够多的运动点得到中位速度方向。
4. 计算与该方向夹角余弦大于 0.7 的运动点比例，并检查速度、方向一致性和连续确认帧数。

触觉示例阈值为至少 6 个运动轨迹、速度 ≥8 ROI 像素/秒、一致性 ≥0.65、持续 3 帧，并有 0.3 秒启动等待。接触能量过低或降至上一帧的 0.85 以下时抑制报警；重复帧不累计证据，长时间缺帧重置历史。local-normal 冷启动门控也作用于该分支。

这些条件支持判断“接触区域出现一致运动”，**不能单独证明真实滑移**。弹性剪切也可能触发，无纹理接触可能漏检；像素速度不能直接当作物体 mm/s。日志提供 `speed_px_s`、`coherence`、`tracks`、`confirm_frames` 和抑制原因，供算法离线复核。

运动停滞与触觉分支采用 OR 聚合。事件分别标记 `state_action_stuck`、`tactile_deformation`、`tactile_slip`，可同时出现；是否执行动作仍由回退与控制仲裁决定。

## 回退与人类介入

控制模式包括 `policy`、`human`、`recovery`、`recovery_hold`、`correction` 和 `paused`。

普通 SpaceMouse 模式下，移动设备或按夹爪键可从策略、回退、保持或自动修正中接管。人工松手后保持，按 `p` 交回策略；若轴或按钮仍有输入则拒绝恢复。`b` 可请求手动回退（需配置启用），`r` 结束本轮并重置，`q` 停止退出。

自动回退使用**实际执行历史中的实测位姿**。`before_last_moving_chunk` 模式找到最后发生实际移动的 chunk N，选择 N−1 开始时的实测位姿；并不重放模型尚未执行的预测。目标过期、距离过大或旋转过大时拒绝回退并保持，不能裁剪成另一个目标。普通回退保持当前夹爪命令。`limited` 模式逐步接近目标，`inference` 模式沿普通推理下发路径发送目标，两者不能混同。

键盘 HIL 是另一条显式流程：第一次 `h` 请求历史回退，到位或失败后保持；第二次 `h` 才进入人工模式，且须让 SpaceMouse 先归零。回退期间的 `h` 被忽略，连续重复按键有防抖。人工模式下空格或 `p` 恢复。该模式还可识别历史夹爪闭合，先恢复闭合前宽度再回退 TCP；它与普通保持夹爪的回退不同。

可选腕部修正服务在回退后建立 anchor；人工修正并恢复时生成监督样本。自动修正还需显式启用、服务就绪，并通过 anchor 漂移、目标范围等校验。无效目标被拒绝，人工接管优先。服务端实现见 [Wrist Correction](https://github.com/neoteai/wrist-correction)。

## 给算法开发者的数据接口

每轮先写入 `pending_*`，结束后确认保留或丢弃，转为 `kept_*` 或 `discarded_*`。

| 文件 | 内容与用途 |
|---|---|
| `manifest.json` | episode 元数据、状态与汇总 |
| `records.msgpack`、`blobs/` | 完整事件与去重的观测载荷 |
| `control.msgpack` | 每个控制周期的紧凑记录，用于检测和执行分析 |
| `pairs.msgpack` | 有效人工动作与时间对齐的影子策略候选 |
| `execution_segments.msgpack` | 执行片段及逐步有效性标记 |

配对只使用实际人工控制且存在有效策略候选的周期；回退、保持和缺失候选不能当作人工偏好标签。使用 `pair_valid` 等有效性字段，不要把补齐或无效步当作监督。request ID、chunk index、generation 和观测/发布时刻用于追溯动作来源；原始预测、平滑候选、下发命令和实测反馈代表不同量。

`decide(False)` 会删除该轮记录和观测载荷，只保留丢弃元数据。独立提交的修正样本和正常触觉示例不随 episode 丢弃而撤销。记录格式和配对生成以 `HIL/recording.py` 及对应 OpenPI 实现为准；EXPO/DSRL 数据需使用各自模块的约定。

## 算法验证建议

先使用离线测试和已记录控制流复核触发条件，再开展受控实验。固定主策略 checkpoint、推理参数、执行频率、初始姿态和物体条件，分别比较：无检测、仅状态反馈、仅形变、仅光流、OR 融合。先以关闭自动回退的监测模式标注误报/漏报，再评估回退和人工介入效果。

分别统计真实卡滞/滑移事件的检出率、无故触发率、检测延迟、控制周期耗时、人工介入次数和任务成功率。应分开分析接触释放、弹性剪切、低纹理与 stale frame；滑动指标改善不等于任务成功率提高。

运行现场配置、令牌、模型、日志、采集数据和备份不纳入本分支。算法改动应同时检查两套 HIL 实现，并保留不会打开硬件的回归验证。

## 当前验证状态

发布隔离副本的离线测试为 **353 通过、13 失败**；13 项均可在原始源码复现，涉及 Aloha、旧 DSRL 协议测试及缺失的相机可视化模块。三个代表性 HIL 配置的 `--check-only` 通过，没有运行真实机器人。详细范围见 [发布验证](docs/VALIDATION.md)。
