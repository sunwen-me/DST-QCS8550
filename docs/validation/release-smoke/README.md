# 分享包独立检查 · 2026-10-31

- 目录独立于原MID工作区，没有依赖Hydra的构建链接。
- RBT SDK0.8.0主机Release构建成功，5/5 CTests通过。
- 主机README CPU/selfsup合成入口完整运行，无预训练网络和先验，2帧窗口约5.62秒。
  合成真值对照0.277°/3.07cm；只验证此合成输入，不推广为实机精度。
- AidLux在新建隔离目录编译libdst_qnn.so，9个发布payload哈希全部通过。
- 实际加载QAIRT2.50/HTPv73并运行deployment-v2，4次evaluator调用，
  本次小样本p50约7.16ms，不是严谨基准，不含几何/优化；worker总耗时约4.38秒。
- QNN合成试验显式以合成真值作先验，仅用于运行链路检查。几何保护拒绝更新，
  prior_retained；零误差来自输入就是真值，不是网络实现零误差标定！
- 相关YAML为实际输出；不比较selfsup/cpu与fast/qnn的worker耗时作为加速倍率。
- 未重启原服务、未运行实时RBT采集、未发布TF、未验收真实设备准确率。

最终主机Python回归113/113通过，包含KITTI深度/确定性与两轮微型训练验证，
不是只检查训练入口--help。5/5 CTests通过。Git LFS共9个payload，均校验SHA256/字节数。
