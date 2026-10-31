# DST · QCS8550 / AidLux / RBT

**只分享 DST：LiDAR–Camera 外参标定独立迁移包，附训练权重、ONNX 与 QCS8550 NPU context。**
不是整个 MID/Hydra 工程，不包含其它 SLAM 项目或设备驱动。
基于 https://github.com/sunwen-me/DST 的 AI 辅助复现，关联论文 https://arxiv.org/abs/2601.01188 。
**不是论文作者官方实现，不宣称达到论文精度。**

> 实验性工程分享：HTP 执行已有验证，但近零漂移、部分平移退化和多步 CPU/NPU 差异仍存在。
> 真实 Mid-360/Gemini 精度未验收。默认只保存候选，不发布外参或 TF。
> 保留先验、Chamfer 下降和推理变快，都不等于标定成功。

## 下载与安装

    git lfs install
    git clone https://github.com/sunwen-me/DST-QCS8550.git
    cd DST-QCS8550
    git lfs pull
    python3 scripts/verify_artifacts.py
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

权重和合成窗口由 Git LFS 保存，不能假定 ZIP 包含实际模型。推荐 Python 3.10+。
AidLux ARM64 请安装适配系统和 Python 的 PyTorch CPU wheel，不要照搬 x86/CUDA wheel。
NPU 路径仍需要 Torch 运行几何流程。仓库不分发 RBT 或高通专有 SDK/runtime。

## CPU 离线演示

    python scripts/run_dst_offline.py --input examples/synthetic/window.bin --config examples/synthetic/config.yaml --mode selfsup --backend cpu

这是两帧人工合成窗口，不是 KITTI 或真实机器人数据。结果输出到唯一的 runs/dst-offline.*/extrinsic.yaml。
首次 selfsup 含粗搜索，远慢于一次网络前向。该入口不连接 Composer，不发布 TF，不覆盖已有结果。
examples/synthetic/truth.yaml 仅是合成真值，绝不能用于真实设备先验。

| 模式 | 预训练评估网络 | 实际计算位置 |
|---|---|---|
| selfsup | 不需要，每窗重新优化 | CPU 几何与 Torch autograd |
| fast | 需要 | CPU 或 HTP 评估网络，几何仍 CPU |
| both | 需要 | 评估网络 + CPU 自监督优化 |

## QCS8550 NPU

**原生 QNN，不要求带 QNN EP 的 ONNX Runtime。** 已提供 QAIRT 2.50 / HTP v73 context。
仅加速评估网络，不把投影、几何、Chamfer 或反向传播搬到 NPU。不同 BSP/runtime/固件可能需要重编译。
板端准备 C++20 编译器、OpenCV core 开发包和自己合法获取的 QNN headers/runtime：

    export QNN_INCLUDE_DIR=/your/qairt/include/QNN
    export DST_QNN_RUNTIME=/your/qairt
    bash scripts/build_dst_qnn.sh
    python scripts/run_dst_offline.py --input /your/window.bin --config /your/config.yaml --init-yaml /your/checked-prior.yaml --mode fast --backend qnn

fast/both 要求显式先验，默认开启几何保护。指定 QNN 失败会报错，不静默回退。
auto_backend_approved=false、live_sensor_accuracy_verified=false 保持不变。
历史单网络热推理 p50 5.74ms 不是整个标定耗时，更不是整链路75倍加速。详见 docs/NPU.md。

## 随附模型

| models/dst-calib 下的目录 | 用途 | 边界 |
|---|---|---|
| deployment-v2 | 当前工程微调候选，离线入口默认试用 | 非论文原损失，未通过实机精度验收 |
| kitti-validation-best | 九组实验按保存验证损失选出的旧基线 | 不是按测试集排名，NPU 仍实验性 |
| kitti-1152-seed0 | 1152帧原实验 best.pt / model_final.pt | 供比较，无 QNN context |

models/artifacts.sha256.json 提供字节数与 SHA256。各目录保留训练/导出/验证元数据，
历史绝对路径只是来源记录，不是安装要求。仅加载可信来源的 checkpoint。

## 可选 RBT 组件

离线和 QNN-only 编译不需要 RBT。实时窗口组件需要 RBT SDK 0.8.0、CMake ≥3.22、C++20 和 yaml-cpp。

    export RBT_SDK_ROOT=/your/rbt-sdk-0.8.0
    bash scripts/build_dst_calib.sh
    ctest --test-dir build --output-on-failure
    ./build/dst-calib/bin/dst_calib_rbt_node --config config/dst_calib_rbt/default.yaml --check-config
    bash scripts/run_dst_calib.sh --mode selfsup --backend cpu

也可 rbt --project rbt.dst-calib.toml run。用户提供匹配 types/*.rbt 合约的雷达、对齐深度与内参 provider。
不依赖 Hydra 或 ROS2。公共 profile 均 publish_results:false；实验 NPU profile 需你提供独立检查过的
config/dst_calib_rbt/measured-prior.yaml，不会编造真实机器人外参。

## 文档与共享边界

- docs/INPUTS.md：输入与 DSTWIN01 格式，变换为 p_camera = T_cam_lidar @ p_lidar，平移米。
- docs/TRAINING.md：便携数据准备、训练、工程微调和导出。
- docs/validation/：已有验证证据与局限，不冒充真实设备准确率。
- Python 回归：pip install -r requirements-dev.txt，然后 bash scripts/test.sh -q。
- NOTICE.md：来源和权利状态。未擅自替上游代码/权重添加 MIT/GPL 许可。

不包含 KITTI 原始数据、用户传感器日志、真实设备先验、凭据、高通/RBT SDK。
公开可下载不等于已授予任意商业使用或再许可。整理截至 2026-10-31。
