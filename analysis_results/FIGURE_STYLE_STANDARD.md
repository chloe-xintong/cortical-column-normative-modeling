# Paper figure style standard

All paper figures use the semantic colors and rendering helpers in
`qc/publication_style.py`. White backgrounds, editable type, no decorative
grid, and subject-level statistical units are the defaults.

## Violin plots

Violin plots show independent statistical units, normally one value per
subject. Values derived from vertices, columns, or pairs must first be reduced
within subject. Do not use a violin for fewer than 20 independent values; for
fewer than 15, show all observations with median and IQR instead. For large
cohorts, do not overlay every observation.

Violin bodies must remain slender. The single-distribution default is width
0.38 (acceptable range 0.35–0.45), light-blue solid fill (`#90BEE0`, alpha
0.55), and a slightly thicker dark-blue outline (`#4B74B2`, 1.1 pt). Do not
use hollow violins, excessively wide bodies, or fills and outlines with the
same visual intensity. A single violin must not occupy most of its panel's
horizontal area.

The internal Tukey box is subordinate: width 0.09, white fill, dark-blue edge
at 0.9 pt, whiskers to the most extreme observations within 1.5 IQR, and an
orange (`#FC8C5A`) median at 1.5 pt. Retain small gray boxplot outliers when
all raw points are omitted. In multi-condition violins, conditions retain
their semantic colors, while box edges, whiskers, and medians are neutral dark
gray to avoid reusing orange for two meanings. KDEs are clipped to observed
data limits and use Scott bandwidth multiplied by 0.85.

Standard caption:

> Violin plots show the kernel density distribution across subjects. Internal
> boxes indicate the interquartile range, center lines indicate the median,
> and whiskers extend to the most extreme observations within 1.5 times the
> interquartile range.

If gray outlier points are present, state that they are boxplot outliers; do
not describe them as all individual subjects.

## Aspect ratios and multi-panel allocation

Do not default to square plotting areas, equal-aspect axes, or mechanically
equal panel widths. Dimensions must reflect the data structure, scientific
importance, information density, and label requirements.

- Continuous trajectory or depth panels: width:height 1.4–1.7:1.
- Two curves with uncertainty: 1.4–1.6:1.
- Single vertical violin/box panel: 0.65–0.85:1, narrow and tall.
- Multi-group violin: 1.1–1.5:1, depending on group count.
- Model or bar comparison: 1.2–1.5:1; horizontal bars: 1.4–1.8:1.
- Ordinary scatter: 1.0–1.25:1.
- Brain multi-view panels: 1.4–1.8:1.
- Heatmaps follow their row/column structure rather than forced square axes.

Reserve a true 1:1 aspect for direct geometric correspondence, such as
observed versus predicted values with an identity line, method-agreement
scatterplots, symmetric square matrices, or spatial coordinates requiring
true geometry.

For a primary plot plus two distribution panels, use ratios such as
`[1.5, 0.8, 0.8]`; for a brain/matrix primary panel, `[1.6, 1.0, 1.0]`; and
for a primary curve plus a derived curve, approximately `[1.35, 1.0]`.
Equal widths are justified only when panels have equal scientific status,
graphical type, information density, and comparable label requirements.
Two single-distribution violin panels that differ only in the reported metric
are a canonical equal-width case; make the overall two-panel canvas narrower
(approximately 5.2 × 3.8 inches) instead of widening one distribution panel.
When a trajectory and a subject-level distribution do not require direct
axis-aligned comparison, export them as separate figures: use a wide trajectory
canvas and a narrow, tall violin canvas rather than combining both into one
nominally wide figure.

## Titles

Plot titles use concise sentence case and describe only the scientific content
shown in the axes. Use forms such as `Profile agreement: O2-FS versus O3` or
`Curvature dependence: O2-FS versus O3`. Prefer `versus` in display text;
abbreviated `vs` may remain in filenames and variable names.

Do not build titles by joining metadata with dots, bullets, pipes, or repeated
subtitles. Cohort definitions, maximum-available versus matched sampling,
sample size, QC status, figure/supplement labels, statistical methods, and
long numerical results belong in the surrounding section or caption rather
than the plot title. Do not add panel letters. Titles are regular weight,
approximately 9 pt at final size; avoid oversized bold supertitles and excess
top whitespace. Anatomical view labels use readable forms such as `Left
lateral` rather than internal codes such as `LH lateral`.

## Legends

Legend frames depend on placement and background; figures do not force every
legend to use the same frame treatment.

- Legends in white space or outside the axes have no frame.
- Legends inside an axes that overlap curves, confidence bands, heatmaps, or
  other data use a white, 90%-opaque background with a thin light-gray border.
- Frames exist only to preserve readability. Do not use thick black borders,
  shadows, colored backgrounds, or excessive padding.
- Omit redundant legends when axis/category labels or direct labels already
  identify the data. Continuous heatmaps and brain surfaces use colorbars
  instead of ordinary legends.

Use 7.5-pt legend text, `handlelength=1.8`, `handletextpad=.5`, and
`labelspacing=.35`. For framed legends, use `edgecolor="#D0D0D0"`,
`framealpha=.90`, `fancybox=False`, `borderpad=.35`, and a 0.6-pt frame.

## Sample-size annotations

Report sample size in the figure caption by default, not inside the panel.

- When all panels use the same cohort, report the sample size once in the
  caption. Do not repeat it in titles, axis labels, violin categories, or
  panels.
- Retain an in-panel sample size only when the effective sample differs across
  groups, panels, conditions, distance bins, or time points. Use small dark-gray
  text without a box or emphasis color.
- Use lowercase $n$ for the analyzed sample. Use both $N$ and $n$ only when
  distinguishing the full cohort from the analyzed subset.
- Bootstrap repetitions are not sample size. State them separately, for
  example: `$n=196$ participants; 10,000 subject-level bootstrap resamples.`
