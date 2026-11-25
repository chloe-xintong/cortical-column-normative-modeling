"""Warped young-reference perturbation and older-CN calibration."""
import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_results.figure6.figure6_new.run_young_reference as base


def inputs():
    flow, order, nodes, depths, roster = [
        pd.read_csv(path, dtype={"subject_id": str}) for path in (
            base.FIGURE7_ROOT / "inputs/cohort_flow_coverage95_n313.csv",
            base.FEATURE_ROOT / "subject_order.csv",
            base.FIGURE6_ROOT / "ico4_node_metadata.csv",
            base.FIGURE6_ROOT / "ico4_node_depth_metadata.csv",
            base.FIGURE7_ROOT / "inputs/final_analysis_roster.csv",
        )
    ]
    md, thick, md7, thick7 = [
        np.load(path, mmap_mode="r", allow_pickle=False) for path in (
            base.FEATURE_ROOT / "matrices/X_ico4_md.npy",
            base.FIGURE6_ROOT / "inputs/X_ico4_local_freesurfer_thickness.npy",
            base.FIGURE7_ROOT / "inputs/X_figure7_ico4_md.npy",
            base.FIGURE7_ROOT / "inputs/X_figure7_local_freesurfer_thickness.npy",
        )
    ]
    base.validate_inputs(flow, order, md, thick, nodes, depths, roster, md7, thick7)
    young, targets, _ = base.build_rosters(flow, order, roster)
    return young, targets, depths, md, thick, md7, thick7


def median(y, sex, thickness, anchor, notes):
    from pcntoolkit import BLR
    from xarray import DataArray

    tm, ts = thickness.mean(), thickness.std()
    ym, ys = y.mean(), y.std()
    if min(ts, ys) <= 0:
        raise ValueError("constant training response or thickness")
    x = np.column_stack((sex, (thickness - tm) / ts))
    test = np.array([[anchor[0], (anchor[1] - tm) / ts]])
    model = BLR(name="young_delta", heteroskedastic=False, ard=False,
                optimizer="l-bfgs-b", n_iter=100, tol=1e-3,
                warp_name="WarpSinhArcsinh", warp_reparam=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model.fit(DataArray(x, dims=("observations", "covariates")),
                      DataArray(np.zeros((len(y), 1), int),
                                dims=("observations", "batch_effect_dims")),
                      {"dummy_batch_effect": {"0": 0}},
                      DataArray((y - ym) / ys, dims=("observations",)))
        finally:
            notes.update(f"warning:{w.category.__name__}:{str(w.message)[:160]}" for w in caught)
    if type(model.warp).__name__ != "WarpSinhArcsinh" or model.heteroskedastic:
        raise TypeError("Unexpected BLR model family")
    latent, variance = model.ys_s2(test, np.zeros((1, 1), int))
    predicted, _ = model.warp.warp_predictions(latent, variance, model.gamma,
                                               percentiles=[.05, .95])
    result = float(np.asarray(predicted).item() * ys + ym)
    if not np.isfinite(result) or not np.isfinite(model.hyp).all() or not np.isfinite(model.nlZ):
        raise ValueError("nonfinite warped fit")
    return result


def node_fit(node, data):
    young, targets, _, md, thick, md7, thick7 = data
    sl = slice(node * base.N_DEPTHS, (node + 1) * base.N_DEPTHS)
    rows = young.matrix_row_index.to_numpy(int)
    sy = young.SUBJECT_SEX.eq(2).to_numpy(float)
    so = targets.SUBJECT_SEX.eq(2).to_numpy(float)
    ty = thick[rows, node]
    to = base.extract_target_node(thick, thick7, targets, node, True)
    yo = base.extract_target_node(md, md7, targets, node, False)
    delta = np.full(yo.shape, np.nan)
    notes = Counter()
    for d, y in enumerate(md[rows, sl].T):
        good = np.isfinite(y) & np.isfinite(ty)
        yg, sg, tg = y[good], sy[good], ty[good]
        if good.sum() < 10 or np.linalg.matrix_rank(
            np.column_stack((np.ones(len(yg)), sg, tg))
        ) != 3:
            notes["invalid_reference_design"] += 1
            continue
        anchor = sg.mean(), tg.mean()
        try:
            reference = median(yg, sg, tg, anchor, notes)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            notes[f"reference_failure:{type(exc).__name__}:{str(exc)[:120]}"] += 1
            continue
        valid = np.isfinite(yo[:, d]) & np.isfinite(to)
        notes["missing_target_input"] += int((~valid).sum())
        for i in np.flatnonzero(valid):
            try:
                updated = median(np.r_[yg, yo[i, d]], np.r_[sg, so[i]],
                                 np.r_[tg, to[i]], anchor, notes)
                delta[i, d] = updated - reference
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                notes[f"augmented_failure:{type(exc).__name__}:{str(exc)[:120]}"] += 1
    return delta, to, notes


def shard(root, node, data):
    targets = data[1]
    path = root / "shards" / f"{node:04d}.npz"
    delta, thickness, notes = node_fit(node, data)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, delta=delta, thickness=thickness, node=node,
                        ids=targets.subject_id.to_numpy(str), qc=json.dumps(notes))
    temporary.replace(path)
    return int(np.isfinite(delta).sum())


def merge(root, data):
    targets, depths = data[1:3]
    ids = targets.subject_id.to_numpy(str)
    delta = np.lib.format.open_memmap(root / "delta_md.npy", mode="w+", dtype="float64",
                                    shape=(len(ids), base.N_NODES, base.N_DEPTHS))
    thickness = np.empty((len(ids), base.N_NODES))
    reports = []
    for node in range(base.N_NODES):
        with np.load(root / "shards" / f"{node:04d}.npz", allow_pickle=False) as result:
            values = result["delta"]
            if (result["node"].item() != node or not np.array_equal(result["ids"], ids)
                    or values.shape != (len(ids), base.N_DEPTHS)
                    or result["thickness"].shape != (len(ids),) or np.isinf(values).any()):
                raise ValueError(f"Invalid shard for node {node}")
            delta[:, node] = values
            thickness[:, node] = result["thickness"]
            reports.append({"node_index": node, "valid": int(np.isfinite(values).sum()),
                            "events": json.loads(result["qc"].item())})
    delta.flush()
    if not any(row["valid"] for row in reports):
        raise ValueError("All merged values are missing")
    np.save(root / "target_thickness.npy", thickness)
    targets.to_csv(root / "participants.csv", index=False)
    depths.to_csv(root / "positions.csv", index=False)
    base.write_json(root / "merge_report.json", {
        "status": "complete", "nodes": reports,
        "total_values": delta.size, "valid_values": sum(row["valid"] for row in reports),
        "all_missing_nodes": sum(row["valid"] == 0 for row in reports),
    })


def score(y, design, cn):
    valid = np.isfinite(y) & np.isfinite(design).all(axis=1)
    train = valid & cn
    x, response = design[train], y[train]
    n, p = x.shape
    e = np.full(len(y), np.nan)
    z = e.copy()
    if n < max(10, p + 3) or np.linalg.matrix_rank(x) != p:
        return e, z
    inverse = np.linalg.pinv(x)
    beta = inverse @ response
    r = response - x @ beta
    sse = r @ r
    candidates = valid & ~cn
    # Use the observed CN age range at this location.
    candidates &= (design[:, 1] >= x[:, 1].min()) & (design[:, 1] <= x[:, 1].max())
    candidates &= np.isin(design[:, 2], x[:, 2])
    e[candidates] = y[candidates] - design[candidates] @ beta
    if sse > 0:
        z[candidates] = e[candidates] / np.sqrt(sse / (n - p))
    h = np.einsum("ij,ji->i", x, inverse)
    with np.errstate(divide="ignore", invalid="ignore"):
        deleted_error = r / (1 - h)
        deleted_var = (sse - r**2 / (1 - h)) / (n - p - 1)
        deleted_z = deleted_error / np.sqrt(deleted_var)
    index = np.flatnonzero(train)
    usable = (1 - h > 1e-10) & np.isfinite(deleted_error)
    e[index[usable]] = deleted_error[usable]
    usable &= np.isfinite(deleted_z)
    z[index[usable]] = deleted_z[usable]
    return e, z


def second(root):
    if json.loads((root / "merge_report.json").read_text())["status"] != "complete":
        raise ValueError("Merge incomplete")
    delta = np.load(root / "delta_md.npy", mmap_mode="r")
    thickness = np.load(root / "target_thickness.npy", mmap_mode="r")
    subjects = pd.read_csv(root / "participants.csv")
    age = subjects.actual_age.to_numpy(float)
    sex = subjects.SUBJECT_SEX.eq(2).to_numpy(float)
    cn = subjects.diagnosis_at_scan.eq("Normal").to_numpy()
    design = np.column_stack((np.ones(len(age)), (age - 60) / 10, sex))
    summary = []
    for label, adjust in (("age_sex_thickness", True), ("age_sex", False)):
        extra, z = [
            np.lib.format.open_memmap(root / f"{label}_{metric}.npy", mode="w+",
                                      dtype="float64", shape=delta.shape)
            for metric in ("extra_md", "z")
        ]
        for node in range(delta.shape[1]):
            x = np.column_stack((design, thickness[:, node])) if adjust else design
            for d in range(delta.shape[2]):
                extra[:, node, d], z[:, node, d] = score(delta[:, node, d], x, cn)
        extra.flush()
        z.flush()
        for band, depths in {**base.DEPTH_BANDS, "all": range(base.N_DEPTHS)}.items():
            view = z[:, :, list(depths)]
            finite = np.isfinite(view)
            frame = subjects[["scoring_row_index", "actual_age", "diagnosis_at_scan"]].copy()
            frame["model"], frame["band"] = label, band
            frame["n_candidate"] = view.shape[1] * view.shape[2]
            frame["n_valid"] = finite.sum(axis=(1, 2))
            frame["positive"] = (view > 2).sum(axis=(1, 2))
            frame["negative"] = (view < -2).sum(axis=(1, 2))
            denominator = frame.n_valid.replace(0, np.nan)
            frame["burden"] = (frame.positive + frame.negative) / denominator
            frame["interval_coverage"] = 1 - frame.burden
            summary.append(frame)
        positions = pd.read_csv(root / "positions.csv")
        for group in ("MCI", "AD"):
            values = extra[subjects.diagnosis_at_scan.eq(group)].reshape(-1, len(positions))
            count = np.isfinite(values).sum(axis=0)
            positions[f"{group}_n"] = count
            positions[f"{group}_mean_extra_md"] = np.divide(
                np.nansum(values, axis=0), count,
                out=np.full(len(count), np.nan), where=count > 0)
        positions.to_csv(root / f"{label}_group_effects.csv", index=False)
    results = pd.concat(summary, ignore_index=True)
    results.to_csv(root / "stage2_burden.csv", index=False)
    results.assign(valid_fraction=results.n_valid / results.n_candidate).groupby(
        ["model", "band", "diagnosis_at_scan"]
    )[["valid_fraction", "interval_coverage", "burden"]].median().to_csv(root / "stage2_qc.csv")
    base.write_json(root / "stage2_report.json", {
        "status": "complete", "cn_scoring": "LOO; patient scoring uses full older CN",
        "z_scale": "training residual SD", "burden_threshold": 2,
        "group_effects": "descriptive mean patient excess; no confidence intervals",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("smoke", "shard", "merge", "second"))
    parser.add_argument("root", type=Path)
    parser.add_argument("start", type=int, nargs="?")
    parser.add_argument("stop", type=int, nargs="?")
    args = parser.parse_args()
    if args.action == "second":
        second(args.root)
        return
    data = inputs()
    if args.action == "merge":
        merge(args.root, data)
        return
    from importlib.metadata import version
    if version("pcntoolkit") != "1.3.0":
        raise ValueError("Expected PCNToolkit 1.3.0")
    if args.action == "smoke":
        nodes = pd.read_csv(base.FIGURE6_ROOT / "smoke_nodes.csv").node_index.unique()[:1]
        total = sum(shard(args.root / "smoke", int(node), data) for node in nodes)
        if total == 0:
            raise ValueError("Smoke produced no finite values")
        print(f"Smoke valid values: {total}")
        return
    for node in range(args.start, min(args.stop, base.N_NODES)):
        print(f"Node {node}: {shard(args.root, node, data)} finite values", flush=True)


if __name__ == "__main__":
    main()
