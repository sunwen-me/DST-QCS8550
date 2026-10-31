# DST 部署优化记录 — 2026-10-31

## 结论与实际使用方式

已完成一次独立工程微调、新 QNN 编译、AidLux 原生 HTP 执行、多窗口评估和
保存窗口 worker 回归。**可隔离试用，不是实机标定精度验收。**
默认配置和旧模型不替换；不重启 Composer/相机/雷达/其他代理任务。
本轮没有启动 RBT 节点，没有发布候选外参或 TF。现有全局默认配置并不是
一律禁止发布；仅本轮离线入口和新实验配置保证不发布。

实际建议：
1. 保留人工或独立方法测得的先验；没有可靠先验时先用 selfsup/CPU 离线估计，
   结果仍须投影叠图和独立场景检查，不直接覆盖机器人 TF。
2. 有先验后可显式尝试 deployment-v2、fast/qnn、几何保护；保存候选与诊断，
   不通过则继续用原先验。**拒绝更新不等于完成标定，几何分数下降也不证明平移正确。**
3. 真正下一步是 Mid360/Gemini 的同步真实数据与独立外参参考，检查单位、内参、
   深度对齐、点密度、共视区域和静态非退化场景，再决定是否做目标域微调。
   本次没有这项实机准确率证据，不把 KITTI 结果冒充 Gemini 精度。
4. auto_backend_approved=false、live_sensor_accuracy_verified=false 保持不变。

## 原始模型与工程修改

原始九组训练产物未覆盖。此前按保存验证损失选取的起点为
`/home/sw/DST/runs/kitti_verification/train_1152_seed1/best.pt`（epoch49），
SHA256 `749d68d48d7d15e6a0c6416e731f0c2fa52e0ef5c63f63f261a03547672e49dc`。
仍保留 `models/dst-calib/kitti-validation-best/` 和原部署默认指向。

新增 `deployment_finetune.py` / `scripts/finetune_deployment.py`：
- 精确左乘 correction 标签 `T_true @ inverse(T_current)`，包含旋转后的平移。
- 原生 ceil-stride 点数上限 12000/24000/60000/不限；不是随机改变在线采样算法。
- identity/small/medium/large 等比例扰动；训练集两步 residual replay。
- 固定 BatchNorm 运行统计；AdamW 小学习率、权重继续训练，不恢复优化器状态。
- 归一化 SmoothL1，旋转1度、平移5cm尺度，**工程目标，不是原论文式16**。
- 保留原论文 trainer/loss，不宣称重现了论文精度。

训练/选型：
- full-v1：576帧，train00–08，12epochs，lr2e-5，batch8，seed20261031。
- full-v2-1152：以上一轮 best 起步，1152帧，18epochs，lr1e-5，seed20261103。
- 两轮均仅用 val09 的48帧选 checkpoint。开发集 raw3步轨迹指标为
  三场景×两点密度的 `rotation_deg_mean + translation_m_mean/0.05` 均值。
- 起点6.049353 → v1 1.503565 → v2 **1.331970**；v2选epoch12，
  非最终epoch18（1.400427）。不以 guard 保留先验的比例伪造训练改进。
- 候选先冻结到 `frozen-candidate.json`，随后才做本轮 sequence10 评估；
  sequence10历史上做过其他验证，**不是从未见过的新独立数据集**。

v2 开发集 raw3步均值（角度° / 平移m）：

| 场景 | 12000点 | 全点云 |
|---|---:|---:|
| identity | 0.1928 / 0.0212 | 0.2535 / 0.0238 |
| small | 0.2132 / 0.0202 | 0.2668 / 0.0263 |
| large | 0.5361 / 0.1375 | 0.4758 / 0.0737 |

**近零漂移仍存在；不能宣布完全修复。**

## 模型与板端部署

主机与 AidLux 都有 `models/dst-calib/deployment-v2/`：

| 文件 | SHA256 |
|---|---|
| best.pt | b86e84d128759c25b89642df135d588b0fefc70581a4e7c477743d394fb76a44 |
| eva_sb.onnx | 480011245a63bba106f8b7fe70833dbcdc7b1ae90186a970545b07d519046b8b |
| eva_sb_qairt250.bin | c0ef10b1b14594714dc0e62e3ee9c448e50036b59a8432490b6b60f7243e5453 |

compile j56mdw2yp / link jpexlyovg 均 SUCCESS。QAIRT2.50、QCS8550、
HTP v73，context10,542,928字节。使用原生 QNN 适配库，**无需 ORT QNN EP**。
自适应平均池化改写为等价常数bin矩阵运算，导出3样本误差≤1.79e-7；
不是整套几何/优化都移到NPU。

`export_manifest.json` 已根据 profile/实际执行更新 htp_execution_verified=true；
这不意味着数值/精度验收：auto_backend_approved 与 live_sensor_accuracy_verified 仍false。

## 板端评估（固定候选）

`test10-candidate-board-qnn-windows.json`：48帧、12个不重叠4帧窗口，
seed20261111；三初值场景×12000/全点云×raw/guarded，共144个case。
真值仅构造预先指定的初值与报告误差，不参与 guard 或更新选择。

QNN均值（° / m）：

| 场景 | 点密度 | raw | guarded |
|---|---|---:|---:|
| identity | 12000 | 0.1670 / 0.01085 | 0 / 0（12/12保留初值） |
| identity | 全点云 | 0.2647 / 0.01586 | 0 / 0（12/12保留初值） |
| small | 12000 | 0.1156 / 0.02038 | 0.1649 / 0.01993（保留初值） |
| small | 全点云 | 0.2472 / 0.02125 | 0.1649 / 0.01993（保留初值） |
| large | 12000 | 0.6008 / 0.12708 | 0.6965 / 0.13725 |
| large | 全点云 | 0.4890 / 0.06896 | 0.6596 / 0.08526 |

large初值均值1.6141°/0.17446m。全点云raw平移p95=0.15457m，
仍有1/12平移变差；guarded全点云p95=0.18055m，12个case均未变差，
但均值旋转/平移并不优于raw。12000点guarded仍有3/12平移变差。
因此不能把保护或NPU速度当成准确率保证。

### 同板 CPU/NPU 数值对照

另跑预先指定的**前3个窗口**，36个case，TorchCPU两线程，与完整QNN报告
中的相同窗口、相同初值逐一配对。详见 `cpu-qnn-matched-first3.json`。
guarded零/小扰动因保留初值而完全一致；large全点云guarded最终两端
最大差0.38771°/0.07174m。**后端多步等价性仍未通过**。
不得将3窗口的结论外推为全部12窗口一致。

### 实际 worker 与重复执行

新入口→真实 `dst_worker.py`→YAML，已在板端两帧KITTI fixture验证。
CPU与QNN各3次，固定输入输出矩阵在各后端内完全重复（maxabs=0），
两后端差0.001142°/0.02859mm，**只针对这个fixture**。
初值1.22456°/0.06240m；guarded CPU0.70906°/0.15550m、
QNN0.70989°/0.15549m：**旋转改善，平移反而更差，仍不合格**。
12000点fixture：CPU0.42899°/0.09146m，QNN0.43000°/0.09179m，
平移同样劣于初值。`worker-summary.json`含所有逐次结果。

both/qnn 新模型已走完 fast→PE→最终SB→guard→YAML，
仅2+2+2步缩减自监督配置，**接线冒烟，不是完整收敛或准确率实验**。
无先验selfsup/CPU合成2帧也跑通，0.26989°/0.02932m，
final_cd0.0077671、24.35s；仅合成示例，不与真实KITTI耗时/准确率混比。

## 速度范围

单 evaluator 相同已保存输入、30次热执行：QNN p50 **5.742ms**、p95 6.307ms；
TorchCPU两线程p50 **434.483ms**。约75.7倍仅为网络前向，
**不是整体标定75倍**。QNN不profiling初始化182ms。3个输入Torch主机/板端
参考maxabs≤2.03e-7；QNN旋转向量差随机0.01308°、真实0.00114°/0.00615°，
平移差1.450mm/0.0337mm/0.959mm，仍会经重新投影/选帧放大。

真实guarded两帧worker本轮各3次中位：CPU6.491s，QNN6.195s，
含启动/加载/几何/保护/输出，不含采集。其他代理正运行或构建，
计时未隔离，**不将其视作受控速度验收或固定硬件性能**。

## 安全入口与可复现命令

`scripts/run_dst_offline.py` 不连接Composer，不发布TF；默认selfsup/cpu。
fast/both必须显式给已独立检查的先验，默认开固定支持几何保护，并保存
实际config快照；只有显式 `--allow-unguarded` 才关闭（仅用于消融）。
已有结果/config快照拒绝覆盖；原始config文件不修改。

板端检查已保存KITTI测试窗口（不是实机上线）：

```bash
cd /home/aidlux/rbt-work
/home/aidlux/xr1-rbt-runner-venv/bin/python3 scripts/run_dst_offline.py \
  --input runs/dst-kitti-checkpoint-smoke-20261031/window.bin \
  --config runs/dst-kitti-checkpoint-smoke-20261031/config.yaml \
  --init-yaml runs/dst-kitti-checkpoint-smoke-20261031/prior.yaml \
  --mode fast --backend qnn
```

不指定output自动生成唯一目录；CPU对照用 `--backend cpu`。
**Gemini真实窗口不能套用这个KITTI 1–60m配置**；Gemini用
`config/dst_calib_rbt/guarded-inference.yaml`（0.25–6m）和真实检查过的先验。

新 `config/dst_calib_rbt/deployment-v2-experimental.yaml` 为RBT save-only/qnn
实验profile，publish_results=false，guard开启。它有意指向**未提供**的
`config/dst_calib_rbt/measured-prior.yaml`；先填真实先验或显式override，
不要把KITTI prior复制成机器人先验。此profile仅做check-config，未启动节点。

回滚：继续用原 `default.yaml` / `kitti-validation-best` 即可，旧文件未覆盖。
不通过修改manifest的批准字段来绕过C++安全检查。

## 测试与证据索引

- Python完整选定回归105 passed，`pytest-final.log`；不是准确率证明。
- 原来的build/dst-calib没有CTest注册，旧build/hydra-gnn有过期不可执行路径；
  两次错误检查原样保留在ctest-*-final.log，不声称它们通过。
- 独立 `native-check/` 仅编译采样头文件测试、调用现有节点check-config，
  不重编译共享Hydra树：主机/板端各4/4 passed，ctest-*-isolated.log。
- 模型选择：full-v1/、full-v2-1152/下experiment/progress/split_files/validation JSON。
- 导出/编译：export.log、compile.log、model export_manifest/qairt250_jobs。
- 数值/性能：benchmark/benchmark.json、htp_profile.jsonl。
- 精度：test10-*.json、cpu-qnn-matched-first3.json、worker-summary.json。
- 日志：worker-pipeline-board.log、worker-both-qnn-wiring.log、selfsup-fallback.log。

最终边界：**本轮交付是更好的离线工程候选与可回滚试用路径，
不是可无人监管替换真实外参的标定器。实机准确率需要真实数据另行验收。**

## 补充：相同主机 CPU 的旧/新模型对照

为避免把先前CPU/CUDA差异归因于权重，另补完整48帧12窗口TorchCPU对照。
输入、初值、seed和配置逐项匹配，结果见 host-cpu-old-new-comparison.json。
先前candidate-host-windows.json为CUDA，仅作额外记录，不用于本表。

| 场景 | 点密度 | 路径 | 旧均值 ° / m | 新均值 ° / m |
|---|---|---|---:|---:|
| identity | 12000 | raw | 1.9746 / 0.35908 | 0.1604 / 0.01072 |
| identity | 12000 | guarded | 0.0000 / 0.00000 | 0.0000 / 0.00000 |
| identity | 全点云 | raw | 0.4356 / 0.06568 | 0.2517 / 0.01443 |
| identity | 全点云 | guarded | 0.0000 / 0.00000 | 0.0000 / 0.00000 |
| small | 12000 | raw | 2.0173 / 0.35594 | 0.1739 / 0.02243 |
| small | 12000 | guarded | 0.1649 / 0.01993 | 0.1649 / 0.01993 |
| small | 全点云 | raw | 0.4845 / 0.06183 | 0.2534 / 0.02143 |
| small | 全点云 | guarded | 0.1649 / 0.01993 | 0.1649 / 0.01993 |
| large | 12000 | raw | 2.1315 / 0.34992 | 0.6004 / 0.12926 |
| large | 12000 | guarded | 1.1524 / 0.15911 | 0.7008 / 0.13848 |
| large | 全点云 | raw | 0.4656 / 0.07970 | 0.5401 / 0.06442 |
| large | 全点云 | guarded | 0.4264 / 0.09174 | 0.6448 / 0.08990 |

结论仅限这组已知KITTI数据：稀疏点输入和近零raw漂移明显改善，但不是全面胜出，
guarded大扰动全点云存在退化，近零raw仍会漂。未依据这些结果重新选权重、调阈值或批准上线。
