# KITTI Odometry 训练

本项目预测 LiDAR–Camera 外参修正，不预测车辆里程计轨迹。数据来源为用户
指定的 [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)，
使用其官方公开 `avg-kitti` S3 存储中的灰度双目、Velodyne 与标定文件。

## 本次已完成的结果（2026-10-31）

RTX3060完成100轮训练，训练/验证累计约532秒；按验证损失选择第98轮模型。
序列10独立测试含48帧×8组固定扰动，共384个样本，一次网络修正：

| 平均误差 | 修正前 | 修正后 | 降幅 |
| --- | --- | --- | --- |
| 旋转（SO(3)测地角） | 1.7808° | 0.6906° | 61.2% |
| 平移（修正向量） | 18.21 cm | 12.20 cm | 33.0% |

旋转/平移中位误差为0.6422°/11.03 cm。最佳权重：
`/home/sw/DST/runs/train_eva_kitti/best.pt`；完整测试报告：
`/home/sw/DST/runs/train_eva_kitti/test_metrics.json`。
权重SHA256：`04098782320dd1dd17fd2650d5bc268d7e217e8ffd4e3c762dd59f1ed847a699`。
最终轮断点、所有权重的有限性、报告指纹与CUDA加载推理均已核对。

## 数据与防泄漏

- 训练：序列 **00–08**，每序列均匀抽取 64 帧，共 **576 帧**。
- 验证：序列 **09**，48 帧；固定随机种子及扰动，按验证损失选 `best.pt`。
- 测试：序列 **10**，48 帧；训练结束后才用最佳验证模型测试。
- 这是覆盖序列00–10的**抽样首版**，不是全量 KITTI 训练。
- 原始 `image_0/image_1` 经 StereoSGBM 生成独立米制相机深度，
  没有将同帧 LiDAR 投影伪装为相机深度。此适配使用双目深度，
  不等同于论文的单目深度 + DAR 实验设置。
- `P0` 是左灰度相机内参，双目基线由 `P0/P1` 得到；
  `Tr` 作为 `T_cam_lidar`，每序列独立保存，避免套用室内名义外参。
- 深度范围1–60 m；虚拟相机256×512、f=600。
- 只用 HTTP Range 提取所选ZIP条目，核对CRC，支持缓存续传。
  数据来源、索引、标定SHA256、划分及帧列表记录于
  `runs/kitti_odometry/manifest.json`。
- NPZ的 `stamp=帧号/10` 是名义训练时间标记，不是原始设备时间，
  不用于运动补偿；图像和LiDAR按原始帧编号匹配。

## 复现

```bash
cd /home/sw/DST
CONDA_PY=/home/sw/Software/anaconda3/envs/dstcalib/bin/python \
  bash scripts/train_kitti_odometry.sh
```

首版：SB、AdamW lr=5e-4、weight_decay=1e-4、OneCycleLR、batch=8、
100 epochs，每帧每轮在线增广1次，总计57,600个训练样本；
沿用项目式(13)–(16)损失，每样本2048个LiDAR点。
相对默认200 epochs×32 samples/frame缩减为首版预算。

输出独立保存为 `runs/train_eva_kitti/`，不会自动切换实机在线标定权重。

| 文件 | 用途 |
| --- | --- |
| `best.pt` | 验证损失最优模型，推理用 |
| `checkpoint.pt` | 每轮原子保存模型、优化器、调度器，恢复用 |
| `model_final.pt` | 最后一轮模型 |
| `metrics.jsonl` | 每轮训练/验证损失、角度和平移误差、耗时 |
| `run_config.json` | 参数和完整训练配置 |
| `test_metrics.json` | 序列10独立测试结果，含未修正基线 |

相同配置和总轮数下重跑同一命令，自动从 `checkpoint.pt` 恢复。
改变总轮数会改变OneCycle日程，额外实验请使用新的输出目录。
扩大抽样量请使用新 `KITTI_DATA_DIR`，避免旧帧混入新清单。

```bash
# 扩大数据量：只准备数据，不自动训练。
/home/sw/Software/anaconda3/envs/dstcalib/bin/python scripts/prepare_kitti_odometry.py \
  --out runs/kitti_odometry_larger --train-frames 256 --eval-frames 96

# 独立评估，固定扰动，一个网络修正步骤。
/home/sw/Software/anaconda3/envs/dstcalib/bin/python scripts/evaluate_kitti_odometry.py \
  --checkpoint runs/train_eva_kitti/best.pt \
  --data-dir runs/kitti_odometry/prepared/test/10 \
  --out runs/train_eva_kitti/test_metrics.json
```

角度为SO(3)测地误差，平移为相机坐标系修正向量误差，
不属于KITTI排行榜的轨迹误差。KITTI车载传感器与Mid-360/Gemini
存在域差异，实机部署前需本机采集验证或微调。

## 测试

ROS会自动加载与conda Python不匹配的pytest插件，测试时关闭自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/sw/Software/anaconda3/envs/dstcalib/bin/python -m pytest -q \
  tests/test_kitti.py tests/test_geometry.py tests/test_models.py tests/test_eva_infer.py
```

2026-10-31本机检查：`tests -m 'not slow'` 共88项通过、1项跳过、3项慢测未运行。
真实KITTI帧的CUDA训练、断点加载和前向推理均已验证。

## 扩大数据量的对照实验

保持训练序列00–08、原始序列09验证集和序列10测试集，逐级增加独立训练帧。
三个训练集嵌套包含，新增帧优先填补原帧之间最大的时间间隔
（maximin time-index spacing）；不会把验证/测试帧加入训练。

2026-10-31，两组扩容训练及独立测试均已完成：

| 训练帧数 | 测试旋转均值 | 测试平移均值 | 旋转较基线改善 | 平移较基线改善 |
| ---: | ---: | ---: | ---: | ---: |
| 576 | 0.6906° | 12.20 cm | — | — |
| 1,152 | 0.6619° | 12.08 cm | 4.2% | 1.0% |
| 2,304 | 0.6848° | 11.83 cm | 0.8% | 3.0% |

结论：平均误差有小幅改善，但旋转误差并不随数据量单调下降。
1152帧组的旋转配对差值为−0.0287°，帧级bootstrap95%区间
[−0.0522°, −0.0071°]；两组平移和2304帧组旋转的区间均跨零。
这不是多训练种子检验，也不代表视频帧独立，不能声称已证明稳定收益。
两组最佳验证权重分别为第47轮和第25轮，均已完成7,200个优化步骤。
完整报告见 [comparison.md](runs/kitti_scaling/comparison.md)，
曲线见 [data_scaling.png](runs/kitti_scaling/data_scaling.png)。

| 训练帧数 | 每序列帧数 | 轮数 | 优化步骤 | 增广样本总数 | 输出目录 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 576 | 64 | 100 | 7,200 | 57,600 | `runs/train_eva_kitti/` |
| 1,152 | 128 | 50 | 7,200 | 57,600 | `runs/train_eva_kitti_2x/` |
| 2,304 | 256 | 25 | 7,200 | 57,600 | `runs/train_eva_kitti_4x/` |

均从头初始化，seed=0、batch=8、AdamW及OneCycleLR设置相同，
按验证损失选择最佳权重，再使用原测试集384组固定扰动进行一次修正评估。
这是固定计算预算的数据多样性实验；不是将更大数据集也训练100轮。
`--compact_dataset`仅在相机点云缓存完成后释放源深度/RGB，
输入张量等价性已通过单元测试，便于在本机内存中训练较大数据集。

```bash
cd /home/sw/DST
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/sw/Software/anaconda3/envs/dstcalib/bin/python -u \
  scripts/kitti_scaling_experiment.py --workers 24
```

流水线复用原始下载缓存和已准备帧，自动准备2倍/4倍数据、训练、
独立测试，并生成 `runs/kitti_scaling/comparison.md`、`comparison.json`
及 `data_scaling.png/svg/pdf`。已完成的实验不会重跑；中断训练使用完整断点恢复。

比较报告还保存逐样本误差，并按测试帧配对bootstrap（每帧8组扰动一起重采样）。
该区间只描述当前已训练权重在这些测试帧上的抽样变化，不包含训练种子变化，
也未校正视频相邻帧相关性。仅一个训练种子，且不同轮长导致验证/选模频率不同，
因此结果用于判断探索性趋势，不能视为多种子统计证明。

## 进一步验证：多种子及新增测试帧

`scripts/verify_kitti_scaling.py` 将每种数据规模增加训练seed=1/2，
与已有seed=0共三次独立初始化。每次仍是7,200步、相同OneCycle日程，
验证扰动显式固定为seed=1000003，避免验证样本跟着训练种子改变。

主分析采用固定第7,200步的 `model_final.pt`，消除每轮验证频率不同导致的
最佳权重选模机会差异；`best.pt` 仅作次分析，两者结果都会保留。
同时从序列10增加192个与原48测试帧不重叠的测试帧，每帧8组固定扰动，
共1,536组新测试样本，扰动种子3000019。这些帧不用于训练、验证或调参。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/sw/Software/anaconda3/envs/dstcalib/bin/python -u \
  scripts/verify_kitti_scaling.py
```

输出保存在 `runs/kitti_verification/`。报告各训练种子的均值和样本标准差，
按相同种子配对比较扩容组与基线；小样本t区间只作为描述性不确定性指标。
另做8个连续排序测试帧的块bootstrap敏感性分析，不把增广样本当独立训练重复。
新测试帧仍来自序列10的一次驾驶，因此这不是跨序列/跨设备泛化证明。

### 2026-10-31验证结果

新增六次训练均已完成，与原seed=0合计九次训练；每次7,200步。
全部18份最终/最佳验证权重均完成原384组及新增1,536组扰动测试。
下表为新增192测试帧上的最终权重结果，均值±训练种子样本标准差：

| 训练帧数 | 旋转误差 | 平移误差 | 旋转/平移改善种子数 |
| ---: | ---: | ---: | ---: |
| 576 | 0.6789° ± 0.0245° | 12.01 ± 0.18 cm | 基线 |
| 1,152 | 0.6635° ± 0.0304° | 11.73 ± 0.17 cm | 3/3、3/3 |
| 2,304 | 0.6720° ± 0.0058° | 11.99 ± 0.27 cm | 1/3、1/3 |

1152帧的旋转和平移均值分别改善约2.3%，三个种子方向一致；
2304帧的收益不稳定，不能支持继续增加帧数就会稳定改善。
新增测试的主分析配对种子95%区间均跨零，三个种子仍不足以证明显著提升。
这与原48帧复核中1152帧方案的两个指标均3/3改善的方向一致。

完整结果：`runs/kitti_verification/report.md`、`summary.json`；
图：`runs/kitti_verification/verification.png/svg/pdf`；
逐样本误差与模型/帧指纹：`errors_*.npz`及`errors_*_provenance.json`。
当前完整回归90项通过、1项跳过、3项慢测未运行；原始模型指纹保持不变。
