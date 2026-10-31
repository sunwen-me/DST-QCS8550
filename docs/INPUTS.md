# 输入与保存窗口

- LiDAR 点云 XYZ 米，RBT 使用 Mid360PointCloud 合约。
- Depth16 必须对齐 Color CameraInfo 尺寸，header unit_scale 换算米。
- 必须真实同步，RBT 使用 HostMonotonic，不混用设备/系统时间。
- 不附驱动，provider 匹配 types/mid360.rbt 和 types/orbbec.rbt。
- 按实际相机设深度范围，不将 KITTI 1–60m 用于室内 Gemini。
- 不得用同帧 LiDAR 投影伪装成独立深度来证明标定精度。

T_cam_lidar 为4×4，p_camera = T_cam_lidar @ p_lidar，平移米。
先验 YAML 可包含 T_cam_lidar 矩阵，旋转正交且det为正。
synthetic truth 不是任何真实机器人先验。

## DSTWIN01：little-endian，version 1

RBT 保存，Python codec 为 src/dst_calib_rbt/scripts/dst_worker.py::read_window。

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

全窗口共用内参与尺寸，points在LiDAR frame，depth在camera frame。
读取器检查magic/version、截断、尺寸和焦距。仅打开可信、大小受控的输入。
合成窗口用于可移植回归，不是实机准确率证据。
