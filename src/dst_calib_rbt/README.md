# DST RBT component

独立DST迁移，主文档见根目录README.md。不依赖Hydra或ROS2，不分发设备驱动。
依赖RBT SDK0.8.0、C++20和yaml-cpp。main.cpp负责同步窗口，orbbec_input.cpp解码帧，
dst_worker.py运行DST算法。

- selfsup/cpu无需评估网络权重；fast/both需权重，NPU仅加速evaluator。
- 公共profile均publish_results:false，未独立验收不要发布外参或TF。
- 变换p_camera = T_cam_lidar @ p_lidar，平移米。
- 输入Mid360PointCloud、rbt.orbbec.Depth@1与CameraInfo@1；Depth16需对齐Color内参。
- 输出合约types/dst_calib.rbt：Extrinsic、Status、ExtrinsicTf，受配置与验收门控。

输入/NPU/训练/证据见docs/，授权状态见NOTICE.md。
UPSTREAM_*文档保留复现背景，不意味着此包提供ROS节点或旧工作目录。
