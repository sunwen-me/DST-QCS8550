# 来源与权利状态

仅抽取 DST 算法和 RBT/QNN 适配，源项目 https://github.com/sunwen-me/DST ，
基线 commit 917cf8013324eef60782d7496dea59aba7824808。关联论文 https://arxiv.org/abs/2601.01188 。
这是用户自己的 AI 辅助复现，不是论文作者官方实现；论文不等于代码授权。

本次核对上游 GitHub license 字段为空，本地未发现上游 LICENSE。
本次不臆造 MIT/GPL、不替第三方再许可、不宣称所有内容已获无限制授权。
新增工程代码也暂未指定统一开源许可证。需要明确的修改、再分发和商业授权时，
应由权利人确定源码与模型许可。依赖库仍遵守各自许可证。

## 模型与数据

models/dst-calib 是本项目现有训练、工程微调与导出产物，不是论文官方权重。
kitti-1152-seed0、kitti-validation-best 使用 KITTI 抽样训练；deployment-v2 为旧候选继续微调。
元数据保存在对应目录，不附 KITTI 原始样本。KITTI 使用条款由使用者自行阅读遵守：
https://www.cvlibs.net/datasets/kitti/ 。本仓库不替 KITTI 权利人授予许可，
公开状态不能说明数据/衍生模型可用于任意商业用途，确定模型授权时还需核对训练数据限制。

QNN context 是本项目模型编译产物，不包含 QNN/QAIRT 头文件、运行库和 DSP 固件。
高通工具链及编译产物仍受对应协议约束。RBT SDK 同样由用户自行合法获取。
examples/synthetic 只含人工合成几何，不含 KITTI 图像/点云。

历史实验 JSON 保留本地路径、固定种子、云编译 job ID 与 SHA256 供溯源，
不是密钥和运行要求。未附 SSH 配置、访问令牌或真实设备先验。
未经独立验证，不用于安全关键在线外参替换。
