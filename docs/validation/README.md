# 验证证据与边界

这里是整理前已有的DST实验记录，主要日期2026-10-31。
report.md的/home/sw/Dev/mid、/home/aidlux/rbt-work、runs/...是历史路径，未附所有运行目录。
旧报告称原全局配置可能发布；本公开版RBT profile已全部设publish_results:false。

- report.md：训练选型、HTP执行、退化与判断。
- host-cpu-old-new-comparison.json：主机新旧模型。
- cpu-qnn-matched-first3.json：仅前3窗口CPU/QNN配对，不是全12窗口CPU验收。
- worker-summary.json：保存窗口重复运行。
- frozen-candidate.json：候选冻结记录。
- benchmark.json：历史单网络性能，不是全流程耗时。

HTP实际运行已验证，但多步数值/准确率未全面验收，真实Mid360/Gemini精度未验收。
分享包另行检查独立构建、Python回归和合成入口，不启动旧服务或替换TF。
