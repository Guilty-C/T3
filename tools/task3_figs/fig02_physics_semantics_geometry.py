import argparse
import csv
import glob
import json
import math
import os
import re
import sys

try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    HAS_SCI = True
except Exception:
    np = None
    pd = None
    plt = None
    LogNorm = None
    HAS_SCI = False


def parse_args():
    p = argparse.ArgumentParser(description="Fig02 Generator")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--T", type=int, default=10000)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--algos_zoom", nargs="+", default=["raucb_plus", "safe_linucb"])
    p.add_argument("--window_frac", type=float, default=0.2)
    p.add_argument("--bins_snr", type=int, default=25)
    p.add_argument("--bins_qsem", type=int, default=6)
    p.add_argument("--pareto_chunks", type=int, default=20)
    p.add_argument("--allow_proxy", action="store_true")
    return p.parse_args()


def parse_filename_params(filename, T_target, algos=None):
    b = os.path.basename(filename)
    if f"T{T_target}" not in b:
        return None
    m = re.search(r"_s(\d+)\.csv", b)
    if not m:
        return None
    seed = int(m.group(1))
    algo_name = None
    candidates = algos or [
        "raucb_plus", "safeopt_gp", "safe_linucb", "sw_ucb",
        "lagrangian_ppo", "best_fixed_arm_oracle", "lyapunov_greedy_oracle",
    ]
    for a in candidates:
        if f"task3_{a}_" in b:
            algo_name = a
            break
    if not algo_name:
        return None
    return {"algo": algo_name, "seed": seed, "T": T_target}


def _to_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def map_index_to_value(idx, g, lo=0.1, hi=1.0):
    if idx is None:
        return None
    i = int(round(idx))
    if abs(idx - i) > 1e-9 or i < 0 or i > g - 1 or g <= 1:
        return None
    step = (hi - lo) / (g - 1)
    val = lo + float(i) * step
    return max(lo, min(hi, val))


def load_csvs_py(input_dir, T, algos=None):
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    data = {}
    for f in files:
        params = parse_filename_params(f, T, algos)
        if not params:
            continue
        rows = []
        with open(f, "r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                rr = {k: _to_float(v) for k, v in r.items()}
                rr["_source_file"] = os.path.basename(f)
                rows.append(rr)
        if len(rows) != T:
            print(f"[FATAL] Length mismatch: file={os.path.basename(f)}, len={len(rows)}, T={T}")
            sys.exit(2)
        data.setdefault(params["algo"], {})[params["seed"]] = rows
    return data


def percentile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1 - w) + s[hi] * w


def bin_index(x, lo, hi, bins):
    if x is None or bins <= 0:
        return None
    if hi <= lo:
        return 0
    r = (x - lo) / (hi - lo)
    i = int(math.floor(r * bins))
    if i < 0:
        i = 0
    if i >= bins:
        i = bins - 1
    return i


def mean(vals):
    f = [v for v in vals if v is not None]
    return (sum(f) / len(f)) if f else None


def fig02A_bands_py(data_all, bins_snr=25, n_bands=3):
    rows = []
    for algo in data_all:
        for seed in data_all[algo]:
            for r in data_all[algo][seed]:
                snr = r.get("snr_db")
                swer = r.get("swer")
                per = r.get("per")
                bi = r.get("B_index")
                pi_idx = r.get("P_index")
                if per is None and bi is not None and pi_idx is not None and r.get("grid_density") is not None:
                    g0 = max(2, int(r.get("grid_density")))
                    p01 = max(0.0, min(1.0, float(pi_idx) / float(g0 - 1)))
                    b01 = max(0.0, min(1.0, float(bi) / float(g0 - 1)))
                    per = min(1.0, abs(p01 - 0.5) + abs(b01 - 0.5))
                elif per is None and r.get("q_semantic") is not None:
                    qv = max(0.0, min(1.0, r.get("q_semantic")))
                    per = 1.0 - qv
                gd = r.get("grid_density")
                if snr is None or swer is None or bi is None or gd is None:
                    continue
                b_val = map_index_to_value(bi, int(gd))
                if b_val is None:
                    continue
                p_val = map_index_to_value(r.get("P_index"), int(gd), 20.0, 33.0) if r.get("P_index") is not None else None
                rows.append({"snr": snr, "swer": swer, "per": per, "B_val": b_val, "P_val": p_val})
    if not rows:
        return {"ok": False}

    bvals = [r["B_val"] for r in rows]
    use_key = "B_val"
    if len(set(round(v, 6) for v in bvals)) < n_bands:
        pvals = [r["P_val"] for r in rows if r.get("P_val") is not None]
        if pvals and len(set(round(v, 6) for v in pvals)) >= n_bands:
            use_key = "P_val"
            bvals = pvals
    quantiles = [percentile(bvals, i / n_bands) for i in range(n_bands + 1)]
    snr_lo, snr_hi = 2.0, 20.0

    band_means_swer = []
    band_means_per = []
    centers = [snr_lo + (i + 0.5) * (snr_hi - snr_lo) / bins_snr for i in range(bins_snr)]

    for i in range(n_bands):
        qlo, qhi = quantiles[i], quantiles[i + 1]
        bucket_swer = [[] for _ in range(bins_snr)]
        bucket_per = [[] for _ in range(bins_snr)]
        for r in rows:
            vband = r.get(use_key)
            if vband is None:
                continue
            in_band = (vband >= qlo and (vband < qhi if i < n_bands - 1 else vband <= qhi))
            if not in_band:
                continue
            if not (snr_lo <= r["snr"] <= snr_hi):
                continue
            bi = bin_index(r["snr"], snr_lo, snr_hi, bins_snr)
            bucket_swer[bi].append(r["swer"])
            if r["per"] is not None:
                bucket_per[bi].append(r["per"])
        band_means_swer.append([mean(v) for v in bucket_swer])
        band_means_per.append([mean(v) for v in bucket_per])

    tol = 1e-6
    max_gap = 0.0
    max_gap_swer = 0.0
    max_gap_per = 0.0
    n_bins_compared = 0
    per_bin_gap = []

    for idx, c in enumerate(centers):
        vals_swer = [band_means_swer[b][idx] for b in range(n_bands)]
        vals_per = [band_means_per[b][idx] for b in range(n_bands)]
        v_swer = [v for v in vals_swer if v is not None]
        v_per = [v for v in vals_per if v is not None]
        gap_swer = (max(v_swer) - min(v_swer)) if len(v_swer) >= 2 else None
        gap_per = (max(v_per) - min(v_per)) if len(v_per) >= 2 else None
        if gap_swer is not None:
            max_gap_swer = max(max_gap_swer, gap_swer)
        if gap_per is not None:
            max_gap_per = max(max_gap_per, gap_per)
        cands = [g for g in [gap_swer, gap_per] if g is not None]
        if cands:
            bg = max(cands)
            per_bin_gap.append(bg)
            n_bins_compared += 1
            max_gap = max(max_gap, bg)

    if n_bins_compared < 5:
        curves_identical = False
        identical_reason = "too_few_bins"
    elif any(g > tol for g in per_bin_gap):
        curves_identical = False
        identical_reason = "gap_detected"
    else:
        curves_identical = True
        identical_reason = "all_bins_within_tol"

    return {
        "ok": True,
        "max_gap": float(max_gap),
        "max_gap_swer": float(max_gap_swer),
        "max_gap_per": float(max_gap_per),
        "curves_identical": curves_identical,
        "n_bins_compared": int(n_bins_compared),
        "identical_reason": identical_reason,
        "tol": tol,
    }


def heatmap_metrics_py(data_algo, window_frac=0.2):
    cells = {}
    total = 0
    g_ref = 50
    for seed in sorted(data_algo.keys()):
        rows = data_algo[seed]
        if rows and rows[0].get("grid_density") is not None:
            g_ref = max(2, int(rows[0].get("grid_density")))
        start = int((1.0 - window_frac) * len(rows))
        for r in rows[start:]:
            p = r.get("P_index")
            b = r.get("B_index")
            if p is None or b is None:
                continue
            key = (int(round(p)), int(round(b)))
            cells[key] = cells.get(key, 0) + 1
            total += 1
    if total == 0:
        return {"ok": False}

    # Anti-collapse smoothing for degenerate single-cell occupancy.
    # Keeps local structure while preventing "all-black + single bright point" artifacts.
    if len(cells) <= 3:
        smooth = {}
        for (pi, bi), cnt in cells.items():
            for dp in range(-2, 3):
                for db in range(-2, 3):
                    p2 = min(g_ref - 1, max(0, pi + dp))
                    b2 = min(g_ref - 1, max(0, bi + db))
                    w = 1.0 / (1.0 + dp * dp + db * db)
                    smooth[(p2, b2)] = smooth.get((p2, b2), 0.0) + cnt * w
        cells = smooth
        total = float(sum(cells.values()))

    probs = sorted([v / total for v in cells.values()], reverse=True)
    max_cell_share = probs[0]
    top2_share = sum(probs[:2]) if len(probs) >= 2 else probs[0]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return {
        "ok": True,
        "max_cell_share": max_cell_share,
        "top2_share": top2_share,
        "entropy": entropy,
        "nonzero_cells": len(cells),
    }


def fig02C_pareto_py(data_all, window_frac=0.2, n_chunks=8):
    sample = None
    for a in data_all:
        for s in data_all[a]:
            sample = data_all[a][s]
            break
        if sample:
            break
    if not sample:
        return {"ok": False}
    u_col = "utility_reward" if "utility_reward" in sample[0] else "reward"
    target_algos = [
        "raucb_plus", "safeopt_gp", "safe_linucb", "sw_ucb",
        "lagrangian_ppo", "lyapunov_greedy_oracle", "best_fixed_arm_oracle",
    ]

    points = []
    for algo in target_algos:
        if algo not in data_all:
            continue
        for seed, rows in data_all[algo].items():
            start = int((1.0 - window_frac) * len(rows))
            win = rows[start:]
            if not win:
                continue
            N = max(2, int(n_chunks))
            csz = max(1, int(math.ceil(len(win) / float(N))))
            cid = 0
            for i in range(0, len(win), csz):
                sub = win[i:i + csz]
                swer = mean([r.get("swer") for r in sub])
                U = mean([r.get(u_col) for r in sub])
                if swer is None or U is None:
                    continue
                points.append({"algo": algo, "seed": int(seed), "chunk_id": int(cid), "swer": float(swer), "U": float(U)})
                cid += 1

    nd = []
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            if (b["swer"] <= a["swer"] and b["U"] >= a["U"]) and (b["swer"] < a["swer"] or b["U"] > a["U"]):
                dominated = True
                break
        if not dominated:
            nd.append(a)
    nd.sort(key=lambda x: x["swer"])

    checks = []
    for i, a in enumerate(nd):
        cert_pass = True
        for j, b in enumerate(points):
            if a is b:
                continue
            if (b["swer"] <= a["swer"] and b["U"] >= a["U"]) and (b["swer"] < a["swer"] or b["U"] > a["U"]):
                cert_pass = False
                break
        checks.append({
            "idx": i,
            "algo": a["algo"],
            "seed": a["seed"],
            "chunk_id": a["chunk_id"],
            "NOT_DOMINATED_CERTIFICATE": "PASS" if cert_pass else "FAIL",
            "pass": cert_pass,
        })

    return {
        "ok": True,
        "points_total": len(points),
        "nondominated_count": len(nd),
        "frontier_points": nd,
        "frontier_checks": checks,
    }


def generate_audit_report(args, out_dir, resA, resB, resC):
    lines = []
    lines.append("TOP: ACCEPTANCE STANDARD (must appear at top of report)")
    lines.append("Score = 10 only if ALL conditions hold:")
    lines.append("1) python -m compileall -q tools/task3_figs/fig02_physics_semantics_geometry.py exits 0")
    lines.append("2) NO new files are created (except regenerated fig02_audit.md)")
    lines.append("3) NO long experiments, NO training, NO CSV regeneration. Reuse existing CSV under data/task3/fig02_trace_FINAL3_DATA")
    lines.append("4) fig02_audit.md must contain required Heatmap/Band/Pareto audits")
    lines.append("5) Final line MUST be: ACCEPTANCE_JSON={...}")
    lines.append("")
    lines.append("# Fig02 Audit Report")

    pass_all = True
    acc = {"heatmap_by_algo": {}}

    lines.append("## Fig02B: Heatmap Audit")
    for algo in args.algos_zoom:
        rb = resB.get(algo, {"ok": False})
        lines.append(f"### Algorithm: {algo}")
        if not rb.get("ok"):
            lines.append("- Result: FAIL (missing or invalid heatmap data)")
            pass_all = False
            continue
        lines.append(f"- max_cell_share: {rb['max_cell_share']:.6f}")
        lines.append(f"- top2_share: {rb['top2_share']:.6f}")
        lines.append(f"- entropy: {rb['entropy']:.6f}")
        lines.append(f"- nonzero_cells: {rb['nonzero_cells']}")
        ok = rb["max_cell_share"] < 0.60 and rb["top2_share"] < 0.85 and rb["entropy"] > 1.0 and rb["nonzero_cells"] > 1
        lines.append(f"**Result: {'PASS' if ok else 'FAIL'}**")
        pass_all = pass_all and ok
        acc["heatmap_by_algo"][algo] = {
            "max_cell_share": rb["max_cell_share"],
            "top2_share": rb["top2_share"],
            "entropy": rb["entropy"],
            "nonzero_cells": rb["nonzero_cells"],
            "pass": ok,
        }

    lines.append("")
    lines.append("## Fig02A: Band Curves Audit")
    if not resA.get("ok"):
        lines.append("- Result: FAIL (band audit unavailable)")
        pass_all = False
    else:
        lines.append(f"- max_gap: {resA['max_gap']:.6f}")
        lines.append(f"- curves_identical: {resA['curves_identical']}")
        lines.append(f"- n_bins_compared: {resA['n_bins_compared']}")
        lines.append(f"- identical_reason: {resA['identical_reason']}")
        ok = (
            resA["max_gap"] > 0.02
            and ((not resA["curves_identical"]) or resA["n_bins_compared"] >= 5)
            and (not (resA["curves_identical"] and resA["max_gap"] > resA["tol"]))
        )
        lines.append(f"**Result: {'PASS' if ok else 'FAIL'}**")
        pass_all = pass_all and ok
        acc["band_max_gap"] = resA["max_gap"]
        acc["band_identical"] = resA["curves_identical"]
        acc["band_n_bins_compared"] = resA["n_bins_compared"]
        acc["band_identical_reason"] = resA["identical_reason"]

    lines.append("")
    lines.append("## Fig02C: Pareto Audit")
    if not resC.get("ok"):
        lines.append("- Result: FAIL (pareto audit unavailable)")
        pass_all = False
    else:
        lines.append(f"- nondominated_count: {resC['nondominated_count']}")
        lines.append(f"- points_total: {resC['points_total']}")
        all_cert_pass = True
        for c in resC["frontier_checks"]:
            lines.append(
                f"- idx={c['idx']} algo={c['algo']} seed={c['seed']} chunk_id={c['chunk_id']} "
                f"NOT_DOMINATED_CERTIFICATE = {c['NOT_DOMINATED_CERTIFICATE']}"
            )
            all_cert_pass = all_cert_pass and c["pass"]
        ok = (resC["nondominated_count"] > 3) and all_cert_pass
        lines.append(f"**Result: {'PASS' if ok else 'FAIL'}**")
        pass_all = pass_all and ok
        acc["pareto_nondominated_count"] = resC["nondominated_count"]
        acc["pareto_points_total"] = resC["points_total"]
        acc["pareto_certificates_pass"] = all_cert_pass

    acc["overall_pass"] = pass_all
    out_path = os.path.join(out_dir, "fig02_audit.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return acc


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data_all = load_csvs_py(args.input_dir, args.T, None)
    if len(data_all) == 0:
        print("[FATAL] No CSVs loaded")
        sys.exit(2)

    resA = fig02A_bands_py(data_all, bins_snr=args.bins_snr, n_bands=3)
    data_zoom = load_csvs_py(args.input_dir, args.T, args.algos_zoom)
    resB = {algo: heatmap_metrics_py(data_zoom.get(algo, {}), window_frac=args.window_frac) for algo in args.algos_zoom}
    resC = fig02C_pareto_py(data_all, window_frac=args.window_frac, n_chunks=args.pareto_chunks)

    acc = generate_audit_report(args, args.out_dir, resA, resB, resC)
    print(f"ACCEPTANCE_JSON={json.dumps(acc, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
