# DST RBT 组件

组件通过 RBT 收集同步窗口，再由 Python worker 运行 DST。依赖 RBT SDK 0.8.0、
C++20 和 yaml-cpp，可独立于 Hydra 或 ROS2 构建。安装步骤见[主文档](../../README.md)。

## 代码结构

- `main.cpp`：收集并同步传感器窗口。
- `orbbec_input.cpp`：解码深度帧。
- `scripts/dst_worker.py`：运行 DST 算法。

## 接口与配置

- 输入为 `Mid360PointCloud`、`rbt.orbbec.Depth@1` 与 `CameraInfo@1`，由传感器 provider 提供。
  `Depth16` 需对齐 Color 内参。
- 输出合约见 `types/dst_calib.rbt`：`Extrinsic`、`Status` 和 `ExtrinsicTf`；发布受配置与验证状态控制。
- 坐标方向为 `p_camera = T_cam_lidar @ p_lidar`，平移单位米。
- `selfsup/cpu` 逐窗口优化，无需预训练评估网络；`fast` / `both` 使用权重，NPU 加速 evaluator。
- 随附 profile 默认 `publish_results: false`，保存候选结果；实机精度验证状态见主文档。

输入、NPU、训练和验证文档位于 [docs/](../../docs/INPUTS.md)，来源与许可见 [NOTICE.md](../../NOTICE.md)。
`UPSTREAM_*` 文档记录原项目的接口和实现背景，其中的路径与命令对应历史工作区。
