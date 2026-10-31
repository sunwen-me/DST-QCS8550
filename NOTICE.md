# 来源与许可

## 源码

本仓库基于 [sunwen-me/DST](https://github.com/sunwen-me/DST)，
基线 commit 为 `917cf8013324eef60782d7496dea59aba7824808`，包含 DST 算法及 RBT/QNN 适配。
算法参考[论文](https://arxiv.org/abs/2601.01188)，属于独立复现，非论文作者官方实现。

截至 2026-10-31，上游仓库未附 LICENSE，本仓库也暂未指定统一源码或模型许可证。
修改、再分发和商业使用的授权范围需向相应权利人确认；第三方依赖按各自许可证使用。

## 模型与数据

`models/dst-calib` 保存本项目训练、微调和导出的模型：

- `kitti-1152-seed0`、`kitti-validation-best`：使用 KITTI 抽样数据训练。
- `deployment-v2`：在已有模型上继续进行工程微调。

训练元数据保存在各模型目录。KITTI 数据需按[数据集使用条款](https://www.cvlibs.net/datasets/kitti/)
另行获取，模型使用和分发也需考虑训练数据的相关限制。`examples/synthetic` 为人工合成数据。

## SDK 与运行环境

QNN context 为本项目模型的编译产物。QNN/QAIRT 头文件、运行库、DSP 固件及 RBT SDK 需另行获取，
按各自协议使用。仓库不附这些 SDK，也不附 KITTI 原始数据、传感器日志、真实设备先验或访问凭据。

实验 JSON 中的本地路径、固定种子、云编译 job ID 和 SHA256 用于记录模型来源与实验过程。
