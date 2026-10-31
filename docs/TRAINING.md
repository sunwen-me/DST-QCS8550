# 训练与导出

权重不是论文官方权重。KITTI访问使用须遵守条款，不附原始数据。
建议有CUDA的主机训练，CPU可训练但更慢；QNN runtime不做反向传播。

## KITTI抽样首版

    source .venv/bin/activate
    pip install opencv-python-headless
    # 阅读并接受KITTI条款后，显式下载所选ZIP条目
    python tools/upstream/prepare_kitti_odometry.py --out runs/kitti_odometry --train-frames 64 --eval-frames 48 --workers 8
    PYTHON=python DEVICE=cuda bash tools/upstream/train_kitti_odometry.sh

训练脚本不自动下载，缺manifest提示退出。无CUDA明确DEVICE=cpu。
默认train00–08共576帧，val09/test10各48帧；--train-frames 128生成1152帧，
新目录用KITTI_DATA_DIR指定。EPOCHS、SAMPLES_PER_FRAME、NUM_WORKERS、KITTI_TRAIN_OUT可覆盖。
权重按验证损失选型，不可用测试集排名。历史sequence10已多次验证，不可称全新未见测试集。
深度由image_0/1双目生成，不是同帧LiDAR投影，也并非论文单目+DAR完整前端。

## 工程微调、ONNX与QNN

    export PYTHONPATH=$PWD/src/dst_calib_rbt/python
    python src/dst_calib_rbt/scripts/finetune_deployment.py --help
    python src/dst_calib_rbt/scripts/evaluate_deployment.py --help
    python src/dst_calib_rbt/scripts/export_eva_onnx.py --help
    python src/dst_calib_rbt/scripts/compile_qairt250.py --help

工程微调用修正标签、归一化residual loss和冻结BN，不等同论文原目标。
deployment-v2为多阶段已有产物，重跑首版不自动精确重建它。
参数在对应training_experiment.json，原始大数据和中间checkpoint未附。
种子20261103只是整数，不是训练日期。云编译需用户自己的Qualcomm AI Hub权限与协议接受，禁止提交凭据。
UPSTREAM_KITTI_TRAINING.md为历史576帧记录，旧路径/权重非本包默认，不同模型指标不混用。
