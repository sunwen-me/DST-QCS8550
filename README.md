# DST

DST 是一个 LiDAR–Camera 外参标定项目。本仓库基于 [sunwen-me/DST](https://github.com/sunwen-me/DST)，
提供 QCS8550 / AidLux 部署、RBT 组件、训练权重，以及 ONNX 和 QNN HTP 模型。
算法参考[论文](https://arxiv.org/abs/2601.01188)，来源与许可见 [NOTICE.md](NOTICE.md)。

支持 CPU 离线标定和 RBT 实时窗口输入。QNN 在 HTP v73 上运行评估网络，几何计算与自监督优化留在 CPU。

## 下载与安装

    git lfs install
    git clone https://github.com/sunwen-me/DST-QCS8550.git
    cd DST-QCS8550
    git lfs pull
    python3 scripts/verify_artifacts.py
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

推荐 Python 3.10+。权重和合成窗口使用 Git LFS，按上述命令克隆可下载完整文件。
AidLux ARM64 需要匹配系统和 Python 版本的 PyTorch CPU wheel；NPU 路径也使用 Torch 处理几何计算。
RBT SDK 和 QNN runtime 需另行安装。

## CPU 离线演示

    python scripts/run_dst_offline.py --input examples/synthetic/window.bin --config examples/synthetic/config.yaml --mode selfsup --backend cpu

示例包含两帧合成数据，真值保存在 `examples/synthetic/truth.yaml`。
结果写入新建的 `runs/dst-offline.*/extrinsic.yaml`，入口独立运行，已有结果保留。
`selfsup` 包含粗搜索和逐窗口优化，耗时高于单次网络推理。

| 模式 | 预训练评估网络 | 实际计算位置 |
|---|---|---|
| selfsup | 不需要，每窗重新优化 | CPU 几何与 Torch autograd |
| fast | 需要 | CPU 或 HTP 评估网络，几何仍 CPU |
| both | 需要 | 评估网络 + CPU 自监督优化 |

## QCS8550 NPU

使用原生 QNN 适配库，无需安装带 QNN EP 的 ONNX Runtime。随附 context 基于 QAIRT 2.50 / HTP v73，
更换 BSP、runtime 或固件后可能需要重新编译。板端需要 C++20 编译器、OpenCV core 开发包和 QNN headers/runtime：

    export QNN_INCLUDE_DIR=/your/qairt/include/QNN
    export DST_QNN_RUNTIME=/your/qairt
    bash scripts/build_dst_qnn.sh
    python scripts/run_dst_offline.py --input /your/window.bin --config /your/config.yaml --init-yaml /your/checked-prior.yaml --mode fast --backend qnn

`fast` / `both` 需要显式提供先验，默认开启几何保护；QNN 加载或执行失败时直接报错。
已测得评估网络热推理 p50 为 5.74 ms，全流程还包含模型加载、投影和几何计算。
安装与 CPU 对照步骤见 [docs/NPU.md](docs/NPU.md)。

## 随附模型

| `models/dst-calib` 下的目录 | 用途 | 说明 |
|---|---|---|
| deployment-v2 | 离线入口默认模型 | 使用工程微调损失，附 ONNX 与 QNN context |
| kitti-validation-best | 原训练基线 | 从九组实验中按验证损失选择 |
| kitti-1152-seed0 | 1152 帧训练对照 | 包含 best.pt / model_final.pt，无 QNN context |

`models/artifacts.sha256.json` 记录文件大小与 SHA256。各模型目录附训练、导出和验证元数据，
其中的历史绝对路径用于追溯实验，安装时使用当前仓库路径。

## 可选 RBT 组件

离线和 QNN-only 编译不需要 RBT。实时窗口组件需要 RBT SDK 0.8.0、CMake ≥3.22、C++20 和 yaml-cpp。

    export RBT_SDK_ROOT=/your/rbt-sdk-0.8.0
    bash scripts/build_dst_calib.sh
    ctest --test-dir build --output-on-failure
    ./build/dst-calib/bin/dst_calib_rbt_node --config config/dst_calib_rbt/default.yaml --check-config
    bash scripts/run_dst_calib.sh --mode selfsup --backend cpu

也可运行 `rbt --project rbt.dst-calib.toml run`。传感器 provider 需匹配 `types/*.rbt` 中的
点云、对齐深度和相机内参合约，组件可独立于 Hydra 或 ROS2 构建。
实验 NPU profile 从 `config/dst_calib_rbt/measured-prior.yaml` 读取设备先验，需按实际设备填写并检查。

## 验证状态

已完成 Python 回归（113 项）、C++ 测试（5 项）、CPU 合成演示和 AidLux 板端 HTP 执行。
当前仍有近零初值漂移、部分场景平移退化，以及 CPU/QNN 多步结果差异；Mid-360/Gemini 实机精度待验证。
因此默认配置只保存候选结果，`publish_results: false`，不发布外参或 TF。
模型元数据中的 `auto_backend_approved` 和 `live_sensor_accuracy_verified` 均为 `false`。

## 文档

- [输入协议](docs/INPUTS.md)：坐标、单位、同步与 DSTWIN01 窗口格式。
- [训练与导出](docs/TRAINING.md)：KITTI 数据准备、训练、工程微调与模型导出。
- [验证记录](docs/validation/README.md)：实验配置、精度、耗时与已知问题。
- [来源与许可](NOTICE.md)：源码、权重、数据和 SDK 的来源说明。

运行 Python 回归：`pip install -r requirements-dev.txt`，然后执行 `bash scripts/test.sh -q`。

文档更新：2026-10-31。
