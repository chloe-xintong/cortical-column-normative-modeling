# 冻结37名年轻 CN 的 ico4 空间特征

## 冻结配置

- 37名年轻 CN 来自既有冻结 roster；4687个 ico4 anchors，保留 hemisphere、节点顺序、fsaverage full-resolution vertex ID 和 DK 标签。
- 邻居为同半球 full-resolution 有效皮层，不限制 DK。固定使用 `local_K_convergence` 的 pair seed 0，同一 pair 表用于所有受试者，不按个体结果重抽。
- 局部为0–10 mm五个2-mm bin、每 bin K≤300；平台为20–30 mm五个2-mm bin、每 bin K≤300。候选不足时使用全部且不重复抽样。
- 有效性、O3采样、全皮层21-depth均值/协方差匹配及 FWHM=0 acquisition null 均复用正式 Figure 4B 定义。
- pair、node、roster 文件及 SHA-256 见 `frozen_config.json`。pair base seed 为20261119；null base seed 为20261020。

## 特征计算

每名受试者使用100个可复现 null realizations。已有10人直接复用原0–19 realization 级结果，并计算20–99；其余27人计算0–99。所有 realization 使用
`stable_null_seed(20261020, subject_id, replicate)`。

每个 real/null realization 均先在五个远距离 bin 内分别计算 pair-correlation 中位数，再跨五个 bin 取中位数作为自身平台。随后计算 `L_real=r_real-p_real`、`L_null=r_null-p_null` 和 `E_realization=L_real-L_null`；最终 `E_100` 为100次 realization 的中位数。负值保留，缺失保持为缺失。

完整审计通过：37人均有100个连续 replicate；每人最终有4687×5=23,435行；10人各复用20次、27人不复用；旧缓存接受前的 K=300 `r_real` 核对误差为0。

## 50次独立批次检查

固定所有 pairs，分别对 realization 0–49 和50–99取中位数，并在每个 anchor×bin 的共同有效受试者上比较。anchor 级分布中位数如下：

| 距离 | Pearson | Spearman | centered residual SD / between-subject SD | absolute common offset | centered residual SD |
|---|---:|---:|---:|---:|---:|
| 0–2 mm | .9981 | .9772 | .062 | .00170 | .0157 |
| 2–4 mm | .9966 | .9820 | .085 | .00148 | .0127 |
| 4–6 mm | .9958 | .9848 | .093 | .00115 | .0103 |
| 6–8 mm | .9956 | .9858 | .095 | .00099 | .0089 |
| 8–10 mm | .9955 | .9861 | .096 | .00088 | .0080 |

典型位置的共同有效人数中位数为37；个别几何/掩码无效位置人数更少，并按原样保留。最终 `E_100` 有效率从0–2 mm的98.89%到8–10 mm的98.93%，没有用0填补。

entorhinal、temporal pole、inferior temporal 和 cingulate 的50-vs-50结果同样稳定：区域内 anchor 层面 Pearson 中位数约 .9946–.9997，残差/受试者间SD中位数约 .025–.104。这里只报告诊断结果，不据此设可靠性阈值或排除区域。

## 文件

- `CN37_E_spatial_features.parquet`：完整 subject×anchor×5 bin 的 `E_100`、两个50次批次 E、`r_real/p_real/L_real`、有效pair数、候选耗尽和平台完整性。
- `subjects/<ID>/local_null_realizations.parquet`：100次局部 `r_null/p_null/L_null/E_realization` 及 null 有效数量。
- `subjects/<ID>/platform_bin_null_realizations.parquet`：100次、五个平台 bin 的 null correlation 和有效pair数。
- `subjects/<ID>/real_features.parquet`：真实数据特征和有效数量。
- `summary/`：受试者审计、覆盖、anchor 和 hemisphere×DK 的50-vs-50稳定性表。

本轮未拟合年轻参考模型、未计算疾病差异、未继续探索 K，也未自动排除任何 DK 区域。
