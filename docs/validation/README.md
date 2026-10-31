# 验证记录

本目录整理了截至 2026-10-31 的 DST 实验结果，包括训练选型、CPU/QNN 对照和部署测试。

| 文件 | 内容 |
|---|---|
| [report.md](report.md) | 训练与微调过程、板端精度、耗时及已知问题 |
| [host-cpu-old-new-comparison.json](host-cpu-old-new-comparison.json) | 同一主机 CPU 上的新旧模型对照 |
| [cpu-qnn-matched-first3.json](cpu-qnn-matched-first3.json) | 前 3 个窗口的同板 CPU/QNN 配对结果 |
| [worker-summary.json](worker-summary.json) | 保存窗口的重复运行结果 |
| [frozen-candidate.json](frozen-candidate.json) | 评估前冻结的模型与配置 |
| [benchmark.json](benchmark.json) | 单个评估网络的推理耗时 |
| [release-smoke/](release-smoke/README.md) | 独立仓库构建、CPU 合成演示和 AidLux HTP 测试 |

HTP 执行已验证。CPU/QNN 多步结果仍有差异，Mid-360/Gemini 实机精度待验证。
发布版 RBT profile 统一设置 `publish_results: false`，只保存候选结果。

详细报告中的 `/home/sw/Dev/mid`、`/home/aidlux/rbt-work` 和 `runs/...` 为历史实验路径，
相关文件按索引保留在本目录和模型目录；完整运行目录未随仓库发布。
