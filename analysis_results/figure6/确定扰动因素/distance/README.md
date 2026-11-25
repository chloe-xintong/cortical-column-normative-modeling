# Distance pilots

- `candidate_coverage/`：ico4 anchor、full-resolution 邻居的 within/across DK
  候选覆盖统计。
- `local_similarity_stability/`：冻结 37 名年轻 CN 中，5 个固定 seed、嵌套
  K=15/30 的真实 `r_real`、`p_real` 和 `L_real` 抽样稳定性。
- `platform_sampling_sensitivity/`：固定已有局部 K=30 `r_real`，比较远距离
  平台 K=30/100/300 的抽样稳定性及其向 0--10 mm `L_real` 的传播。
- `null_adjusted_E_stability/`：10名冻结年轻 CN、3套 pair seed、20次正式
  Figure 4B acquisition null 的小规模 null-adjusted E 稳定性检查。
  其中 `between_subject_stability/` 只复用现有结果，诊断 pair 选择与两个
  10次 null 批次对受试者差异和排序的影响。
- `scripts/`：上述分析的可复现实现。正式 Figure 4 的公共几何与 profile
  mask 读取函数仅作为只读依赖复用。

本目录没有运行 null、组间检验、年轻参考模型或扰动网络。
