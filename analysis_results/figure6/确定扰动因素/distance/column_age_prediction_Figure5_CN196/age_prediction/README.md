# Column局部相似性与平台的年龄预测价值

## 队列与特征

使用 Figure 5 正式196名 CN 的原始 `subject_order` 和 nested 5×5 folds。原元数据候选为200人；D0072、D0094、D0191、D0222 因 Figure 5 profile coverage `<0.95` 未进入正式196人。当前冻结37名年轻 CN 全部属于这196人；其余159人按同一配置补算，没有用“37名年轻+另一批年长”重新定义队列。

所有196人统一使用 ico4 anchors、同半球 full-resolution邻居、pair seed 0、局部0–10 mm五个bin K≤300、平台20–30 mm五个bin K≤300，以及正式 Figure 4B null定义。每人均为100次null realization。负E保留，缺失保持NaN。

四套矩阵严格使用同一subject顺序：

- `E_local`: 196×23,435（4687 anchors×5 bins）
- `Plateau`: 196×4,687（每anchor的 `p_real`）
- `E_local + Plateau`: 196×28,122
- `ico4 MD profile`: 196×98,427（4687×21 depths）

E_local有两个全队列均缺失的0–2 mm特征：左侧fusiform和右侧inferior temporal的fsaverage vertex 11。它们在保存矩阵中保持NaN；正式 `SimpleImputer(strategy="median")` 在每个fold跳过这两个无观测列，没有填0。

## 模型与交叉验证

完全复用 Figure 5 的 LSQR Ridge、21个 `logspace(-4,6,21)` alpha 和原 nested 5×5 folds。每个inner/outer训练fold分别拟合median imputation和standardization。外层fold测试人数为40、39、39、39、39；每人只有一次held-out预测。四模型的受试者、fold标签和年龄逐行一致。

## OOF结果

| 模型 | MAE | RMSE | Pearson r | R² |
|---|---:|---:|---:|---:|
| E_local | 7.400 | 9.109 | .558 | .247 |
| Plateau | 6.883 | 8.549 | .587 | .337 |
| E_local + Plateau | 7.131 | 8.774 | .593 | .302 |
| ico4 MD profile | 5.291 | 6.939 | .761 | .563 |

平台单独优于E_local；联合模型的相关略高于二者，但MAE介于二者之间。原始21-depth MD profile在全部四项指标上最好。

## 配对subject bootstrap

对固定OOF预测的逐受试者绝对误差作10,000次共享受试者重采样（seed 20261028），没有在bootstrap内重新训练。定义 `ΔMAE=联合模型−比较模型`，负值有利于联合模型：

| 比较 | ΔMAE | 95% CI |
|---|---:|---:|
| 联合 vs E_local | −0.268 | [−0.429, −0.104] |
| 联合 vs Plateau | +0.249 | [−0.068, +0.557] |
| 联合 vs MD profile | +1.841 | [+1.312, +2.369] |

因此，加入平台相对E_local降低MAE；但联合模型没有显示优于Plateau，且明显不及原始MD profile。该结论仅涉及CN年龄预测，不包含brain-age gap疾病效应或扰动网络分析。

## 输出

- `features/`: 四模型矩阵、subject顺序及anchor/bin索引。
- `folds/`: 原 Figure 5 outer/inner folds副本。
- `models_lsqr/`: 每模型OOF预测、fold指标、alpha曲线及散点图。
- `summary/`: 统一OOF表、MAE/RMSE/r/R²、共享fold表、配对bootstrap及四面板散点图。
