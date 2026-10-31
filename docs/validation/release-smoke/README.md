# 独立部署测试 · 2026-10-31

## 环境与结果

仓库在独立于原 MID 工作区的目录中构建，使用 RBT SDK 0.8.0；构建链接独立于 Hydra。

| 检查项 | 结果 |
|---|---|
| 主机 Release 构建 | 成功，5/5 CTests 通过 |
| Python 回归 | 113/113 通过，含 KITTI 深度、确定性和两轮微型训练 |
| Git LFS 文件 | 9 个文件的 SHA256 和字节数均匹配 |
| CPU / selfsup 合成演示 | 2 帧，无预训练网络和先验，约 5.62 s |
| 合成演示误差 | 旋转 0.277°，平移 3.07 cm |
| AidLux QNN 构建 | 在独立目录成功编译 libdst_qnn.so |
| AidLux HTP 执行 | QAIRT 2.50 / HTP v73，deployment-v2，4 次 evaluator 调用 |
| 本次 HTP 网络耗时 | p50 约 7.16 ms，4 次调用的小样本记录 |
| 本次 QNN worker 总耗时 | 约 4.38 s，包含几何计算等流程 |

## 输入与结果说明

CPU 与 QNN 测试均使用合成窗口。QNN 的先验设为合成真值，几何保护拒绝更新，
输出状态为 `prior_retained`。结果的零误差来自保留的真值先验。
两项测试分别使用 `selfsup/cpu` 和 `fast/qnn`，计算过程不同，耗时分别记录。
实际输出保存在同目录 YAML，HTP profile 摘录见 `htp-profile-excerpt.jsonl`。

本次覆盖离线部署与运行链路；实时 RBT 采集、TF 发布及真实设备精度待后续验证。原服务保持运行状态。
