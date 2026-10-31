# 输入与保存窗口

## 传感器输入

- LiDAR 点云为 XYZ，单位米；RBT 使用 `Mid360PointCloud` 合约。
- `Depth16` 对齐 Color `CameraInfo` 的图像尺寸，通过 header 的 `unit_scale` 换算为米。
- 点云与深度需同步，RBT 时间戳统一使用 `HostMonotonic`。
- 传感器驱动由 provider 提供，接口见 `types/mid360.rbt` 和 `types/orbbec.rbt`。
- 深度范围按设备设置：KITTI 示例为 1–60 m，Gemini 配置为 0.25–6 m。
- 标定精度验证使用独立相机深度或外参参考；同帧 LiDAR 投影只用于几何链路测试。

## 坐标约定

`T_cam_lidar` 为 4×4 矩阵，方向为 `p_camera = T_cam_lidar @ p_lidar`，平移单位米。
先验 YAML 可包含 `T_cam_lidar` 矩阵，旋转部分需正交且行列式为正。
真实设备先验由测量或独立标定获取，`examples/synthetic/truth.yaml` 对应合成示例。

## DSTWIN01：little-endian，version 1

RBT 保存此格式的窗口，Python 读取入口为
`src/dst_calib_rbt/scripts/dst_worker.py::read_window`。

| 字段 | 类型 |
|---|---|
| magic | 8 bytes DSTWIN01 |
| version, frame_count, width, height | 各uint32 |
| fx, fy, cx, cy | 各float64 |
| 每帧timestamp_ns | uint64 |
| 每帧point_count | uint32 |
| 每帧XYZ | point_count×3 float32，交错XYZ |
| 每帧depth_count | uint32，等于width×height |
| 每帧depth | float32，row-major，米 |

全窗口共用内参与尺寸，points 位于 LiDAR 坐标系，depth 位于 camera 坐标系。
读取器检查 magic/version、文件截断、尺寸和焦距；输入文件应来自可信来源且大小受控。
合成窗口用于跨平台回归，实机精度验证需另行采集同步数据。
