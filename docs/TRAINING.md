# 训练与导出

建议在 CUDA 主机训练，也支持 CPU；QNN 用于板端推理，训练和反向传播使用 PyTorch。
随附权重由本项目训练，数据与模型来源见 [NOTICE.md](../NOTICE.md)。

## KITTI 抽样训练

    source .venv/bin/activate
    pip install opencv-python-headless
    # 按 KITTI 使用条款下载所选 ZIP 条目
    python tools/upstream/prepare_kitti_odometry.py --out runs/kitti_odometry --train-frames 64 --eval-frames 48 --workers 8
    PYTHON=python DEVICE=cuda bash tools/upstream/train_kitti_odometry.sh

先运行数据准备脚本生成 manifest，再启动训练。CPU 训练时设置 `DEVICE=cpu`。

默认数据划分为 train 00–08（共 576 帧）、val 09（48 帧）和 test 10（48 帧）。
设置 `--train-frames 128` 可生成 1152 帧训练集；使用其他数据目录时设置 `KITTI_DATA_DIR`。
训练轮数、每帧采样数、进程数和输出目录分别由 `EPOCHS`、`SAMPLES_PER_FRAME`、`NUM_WORKERS`
和 `KITTI_TRAIN_OUT` 控制。

Checkpoint 按验证损失选择。Sequence 10 已用于多轮历史验证，属于已使用过的测试数据。
训练深度来自 `image_0/1` 双目图像；这一前端与论文的单目 + DAR 路径不同。

## 工程微调与导出

    export PYTHONPATH=$PWD/src/dst_calib_rbt/python
    python src/dst_calib_rbt/scripts/finetune_deployment.py --help
    python src/dst_calib_rbt/scripts/evaluate_deployment.py --help
    python src/dst_calib_rbt/scripts/export_eva_onnx.py --help
    python src/dst_calib_rbt/scripts/compile_qairt250.py --help

工程微调使用修正标签、归一化 residual loss 和冻结 BatchNorm；原始 trainer/loss 也保留在代码中。
`deployment-v2` 经多阶段微调得到，各阶段参数记录在对应的 `training_experiment.json`。
复现实验需按这些记录依次运行，初版训练脚本生成的是首版模型。仓库保留发布权重及元数据，
原始训练数据和中间 checkpoint 需另行准备。

随机种子按整数解释，例如 `20261103`。QNN 云编译需要配置自己的 Qualcomm AI Hub 账号并接受相关协议，
访问凭据保存在本地环境。

[UPSTREAM_KITTI_TRAINING.md](UPSTREAM_KITTI_TRAINING.md) 保留了历史 576 帧训练过程，
其中的路径和模型名对应当时的实验；当前发布模型见 README 的模型表。
