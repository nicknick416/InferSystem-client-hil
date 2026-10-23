# 发布验证

本分支从 HIL 客户端源码快照整理，保留 VTLA/OpenPI HIL、检测器、回退、记录和测试。现场设备配置替换为脱敏示例；不包含令牌、采集数据、模型、日志或备份。示例默认关闭修正服务及其自动执行；部署时需要显式配置。

2026-10-23，在 Linux 客户端现有依赖环境中的隔离副本运行：

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

结果为 **353 passed，13 failed**。以下 13 项失败均已在未修改的原始源码上复现，未为发布而跳过、删除或改写断言：

| 范围 | 数量 | 现象 |
|---|---:|---|
| `tests/test_aloha_robot.py` | 1 | 笛卡尔控制模式调用与断言不一致 |
| `tests/test_dsrl_actor_worker.py` | 2 | 期望的 `dsrl` 请求字段不存在 |
| `tests/test_pi_dsrl_capture.py` | 5 | 旧 DSRL 记录协议字段与当前 worker 不一致 |
| `tests/test_realsense_visualize.py` | 5 | 源码中缺少 `Example.realsense_visualize` 模块 |

VTLA 基础配置、触觉检测示例和 OpenPI RTC 配置的 `--check-only` 均通过；没有连接机器人、相机、HID 或推理服务，没有执行真实机器人运动。

这不表示完整测试套件通过，也不证明上述可选功能可直接部署。基础 HIL、RTC、检测、人工接管与修正相关离线用例的通过结果不能替代真机延迟、误报/漏报或成功率评估。
