# QCS8550 / AidLux 原生 QNN

仅 fast/both 的 evaluator 在 HTP v73 加速。selfsup、几何和反向传播仍在 CPU。
不依赖 ORT QNN EP；ONNX Runtime 仅供主机导出一致性测试。
用户自行合法获取 QAIRT 2.50 headers/runtime，仓库不分发专有 SDK。

## 当前运行库布局

    /your/qairt/lib/aarch64-oe-linux-gcc11.2/libQnnHtp.so
    /your/qairt/lib/aarch64-oe-linux-gcc11.2/  # 配套 host runtime/stub
    /your/qairt/lib/hexagon-v73/unsigned/     # 配套 skel/DSP

BSP 还需兼容 FastRPC/DSP 设备、驱动和固件。其它布局需在自己的 runtime 目录
建立适当布局或修改 worker/adapter，不能只改 LD_LIBRARY_PATH 就假定兼容。
AArch64 库不可在 x86_64 主机直接运行。

    export QNN_INCLUDE_DIR=/your/qairt/include/QNN
    export DST_QNN_RUNTIME=/your/qairt
    bash scripts/build_dst_qnn.sh

需要 C++20 编译器、pkg-config 和 OpenCV core 开发包，不需要 RBT SDK。
也可 QNN-only CMake：

    cmake -S . -B build-qnn -DDST_BUILD_RBT=OFF -DDST_ENABLE_QNN=ON -DQNN_INCLUDE_DIR=/your/qairt/include/QNN -DCMAKE_BUILD_TYPE=Release
    cmake --build build-qnn --target dst_qnn --parallel 2

默认 Python 查找 build/dst-calib/lib/libdst_qnn.so。如果使用 build-qnn，离线入口传
--qnn-library build-qnn/dst-calib/lib/libdst_qnn.so。
可显式 --qnn-runtime /your/qairt、--qnn-profile runs/profile.json。

输入 NCHW float32 [1,3,256,512]，输出 [1,6] 轴角弧度及米制平移。
加载前检查 context SHA256，输入尺寸与有限性也校验。
context 不保证任意 QCS8550 镜像通用，版本不兼容需重编译并实际验收。

## 试用顺序

1. python scripts/verify_artifacts.py 排除 LFS 指针和下载损坏。
2. CPU 合成 selfsup 演示确认依赖与输入。
3. 自己的保存窗口、正确深度配置、独立检查先验，显式 fast/qnn。
4. 同窗口/先验用 fast/cpu 对照投影、旋转、平移与重复稳定性。
5. 没有实机独立参考，不发布 TF、不修改机器人已有先验。

QNN 指定失败会报错，不静默回退。自动批准/实机精度标志均 false，勿手改 true 绕过门控。
历史单网络5.74ms不包括投影、初始化、点云或优化。验证边界见 docs/validation/。
