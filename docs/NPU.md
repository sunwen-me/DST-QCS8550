# QCS8550 / AidLux NPU 部署

`fast` / `both` 的 evaluator 通过原生 QNN 在 HTP v73 上执行，
`selfsup`、投影、几何计算和反向传播使用 CPU。ONNX Runtime 用于主机端导出一致性测试，
板端无需 ORT QNN EP。

## 环境

需要 QAIRT 2.50 headers/runtime、C++20 编译器、pkg-config 和 OpenCV core 开发包。
QNN-only 构建可独立运行，无需 RBT SDK。SDK 获取与许可见 [NOTICE.md](../NOTICE.md)。

## 当前运行库布局

    /your/qairt/lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so
    /your/qairt/lib/aarch64-oe-linux-gcc11.2/  # 配套 host runtime/stub
    /your/qairt/lib/hexagon-v73/unsigned/     # 配套 skel/DSP

BSP 需要匹配的 FastRPC/DSP 驱动和固件。以上为 AArch64 板端布局；使用其他目录结构时，
需调整 runtime 目录或 worker/adapter 中的查找路径。

## 编译

    export QNN_INCLUDE_DIR=/your/qairt/include/QNN
    export DST_QNN_RUNTIME=/your/qairt
    bash scripts/build_dst_qnn.sh

也可直接使用 CMake：

    cmake -S . -B build-qnn -DDST_BUILD_RBT=OFF -DDST_ENABLE_QNN=ON -DQNN_INCLUDE_DIR=/your/qairt/include/QNN -DCMAKE_BUILD_TYPE=Release
    cmake --build build-qnn --target dst_qnn --parallel 2

默认 Python 查找 `build/dst-calib/lib/libdst_qnn.so`。使用 `build-qnn` 时，
给离线入口传入 `--qnn-library build-qnn/dst-calib/lib/libdst_qnn.so`。
运行库与 profile 路径可通过 `--qnn-runtime /your/qairt` 和 `--qnn-profile runs/profile.json` 指定。

## 运行与对照

1. 执行 `python scripts/verify_artifacts.py`，检查模型文件完整性。
2. 运行 README 中的 CPU 合成示例，确认 Python 环境可用。
3. 准备保存窗口、匹配传感器的深度配置和检查过的设备先验：

       python scripts/run_dst_offline.py --input /your/window.bin --config /your/config.yaml --init-yaml /your/checked-prior.yaml --mode fast --backend qnn

4. 将后端改为 `--backend cpu`，用同一窗口和先验对照投影、旋转、平移及重复运行结果。

`fast` / `both` 默认开启几何保护。QNN 加载或执行失败时直接报错，方便定位环境问题。
更换 BSP、runtime 或固件后，需检查 context 兼容性，必要时重新编译并重复上述对照。

## 模型接口

- 输入：NCHW float32，形状 `[1, 3, 256, 512]`。
- 输出：`[1, 6]`，包括轴角旋转向量（弧度）和平移（米）。
- 加载检查：context SHA256、输入尺寸及数值有限性。

## 实测与待验证项

AidLux QCS8550 已实际加载 QAIRT 2.50 / HTP v73 并运行随附模型。历史评估网络热推理
p50 为 5.74 ms，这一计时范围仅含网络前向。全流程另有初始化、点云、投影与优化开销。

CPU/QNN 多步结果一致性和 Mid-360/Gemini 实机精度仍待验证，
`auto_backend_approved=false`、`live_sensor_accuracy_verified=false`。
默认仅保存候选结果，TF 发布待实机验证后再配置。详细数据见[验证记录](validation/README.md)。
