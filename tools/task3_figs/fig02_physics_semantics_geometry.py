import argparse
import os
import glob
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import json

def parse_args():
    p = argparse.ArgumentParser(description="Fig02 Generator")
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--T", type=int, default=10000)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--algos_zoom", nargs="+", default=["raucb_plus","safe_linucb"])
    p.add_argument("--window_frac", type=float, default=0.2)
    p.add_argument("--bins_snr", type=int, default=25)
    p.add_argument("--bins_qsem", type=int, default=6)
    p.add_argument("--pareto_chunks", type=int, default=1)
    p.add_argument("--allow_proxy", action="store_true", help="Allow q_semantic as proxy if s_t missing")
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
    if algos:
        for a in algos:
            if f"task3_{a}_" in b:
                algo_name = a
                break
    else:
        for a in ["raucb_plus","safeopt_gp","safe_linucb","sw_ucb","lagrangian_ppo","best_fixed_arm_oracle","lyapunov_greedy_oracle"]:
            if f"task3_{a}_" in b:
                algo_name = a
                break
    if not algo_name:
        return None
    return {"algo": algo_name, "seed": seed, "T": T_target}

def load_csvs(input_dir, T, algos=None):
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    data = {}
    for f in files:
        params = parse_filename_params(f, T, algos)
        if not params:
            continue
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[FATAL] CSV read failed: file={os.path.basename(f)}, error={e}")
            sys.exit(2)
            
        if len(df) != T:
            print(f"[FATAL] Length mismatch: file={os.path.basename(f)}, len={len(df)}, T={T}")
            sys.exit(2)
            
        df["_source_file"] = os.path.basename(f)
        data.setdefault(params["algo"], {})[params["seed"]] = df
    return data

def map_index_to_value(idx, g, lo=0.1, hi=1.0):
    if pd.isna(idx) or not np.isfinite(idx):
        return np.nan
    
    if abs(idx - round(idx)) > 1e-9:
        return np.nan
        
    i = int(round(idx))
    if i < 0 or i > g - 1:
        return np.nan
    
    step = (hi - lo) / (g - 1)
    val = lo + float(i) * step
    return max(lo, min(hi, val))

def check_required_columns(data_dict, required_cols, fig_name):
    for algo in data_dict:
        for seed, df in data_dict[algo].items():
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                src = df["_source_file"].iloc[0] if "_source_file" in df.columns else "unknown"
                print(f"[FATAL] Missing columns: {missing} , file={src} , algo={algo}, seed={seed}")
                sys.exit(2)

def fig02A_physics(data_all, out_path, bins_snr=25):
    dfs = []
    for algo in data_all:
        for seed, df in data_all[algo].items():
            dfs.append(df[["snr_db","swer","per"]].copy())
    if len(dfs) == 0:
        return {"ok": False, "msg": "No CSVs for Fig02-A"}
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    snr = df_all["snr_db"].values
    swer = df_all["swer"].values
    per = df_all["per"].values
    bins = np.linspace(np.nanmin(snr), np.nanmax(snr), bins_snr+1)
    centers = 0.5*(bins[:-1] + bins[1:])
    idx = np.digitize(snr, bins) - 1
    idx = np.clip(idx, 0, bins_snr-1)
    counts = np.array([np.sum(idx==i) for i in range(bins_snr)])
    means_swer = np.array([np.nanmean(swer[idx==i]) if np.any(idx==i) else np.nan for i in range(bins_snr)])
    means_per = np.array([np.nanmean(per[idx==i]) if np.any(idx==i) else np.nan for i in range(bins_snr)])
    plt.figure(figsize=(6,4))
    plt.plot(centers, means_swer, label="On-policy average (algos x seeds)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("sWER")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "fig02A_snr_swer.png"), dpi=200)

    plt.figure(figsize=(6,4))
    plt.plot(centers, means_per, label="On-policy average (algos x seeds)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "fig02A_snr_per.png"), dpi=200)
    
    snr_range = np.nanmax(snr) - np.nanmin(snr)
    swer_range = np.nanmax(swer) - np.nanmin(swer)
    per_range = np.nanmax(per) - np.nanmin(per)
    
    warning_lines = []
    
    if snr_range < 1e-6:
        warning_lines.append(f"WARNING: degenerate_snr_range (range={snr_range:.2e})")
    
    if swer_range < 1e-6:
        warning_lines.append(f"WARNING: degenerate_swer_range (range={swer_range:.2e})")

    if per_range < 1e-6:
        warning_lines.append(f"WARNING: degenerate_per_range (range={per_range:.2e})")

    finite_bins_swer = np.where(np.isfinite(means_swer))[0]
    finite_swer = means_swer[np.isfinite(means_swer)]
    finite_bins_per = np.where(np.isfinite(means_per))[0]
    finite_per = means_per[np.isfinite(means_per)]

    if (snr_range < 1e-6) or (swer_range < 1e-6):
        monotonic_swer = "degenerate"
        viol_points_swer = []
    else:
        monotonic_swer = bool(np.all(np.diff(finite_swer) <= 1e-6))
        viol_points_swer = finite_bins_swer[np.where(np.diff(finite_swer) > 1e-6)[0]].tolist()

    if (snr_range < 1e-6) or (per_range < 1e-6):
        monotonic_per = "degenerate"
        viol_points_per = []
    else:
        monotonic_per = bool(np.all(np.diff(finite_per) <= 1e-6))
        viol_points_per = finite_bins_per[np.where(np.diff(finite_per) > 1e-6)[0]].tolist()

    return {
        "ok": True,
        "monotonic": monotonic_swer,
        "viol_points": viol_points_swer,
        "per_monotonic": monotonic_per,
        "per_viol_points": viol_points_per,
        "bins_centers": centers.tolist(),
        "per_bin_counts": counts.tolist(),
        "per_bin_means": [float(v) if np.isfinite(v) else np.nan for v in means_per],
        "warning_lines": warning_lines,
    }

def fig02A_optional_B_bands(data_all, out_path, bins_snr=25, n_bands=3):
    if data_all:
        first_algo = next(iter(data_all))
        if data_all[first_algo]:
            first_seed = next(iter(data_all[first_algo]))
            df = data_all[first_algo][first_seed]
            required = ["snr_db", "swer", "per", "B_index", "grid_density"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                src = df["_source_file"].iloc[0] if "_source_file" in df.columns else "unknown"
                print(f"[FATAL] Missing columns for bands: {missing}, file={src}")
                sys.exit(2)

    dfs = []
    for algo in data_all:
        for seed, df in data_all[algo].items():
            d = df[["snr_db","swer","per","B_index","grid_density"]].copy()
            d["B_val"] = d.apply(lambda r: map_index_to_value(r["B_index"], int(r["grid_density"])), axis=1)
            dfs.append(d)
    if len(dfs) == 0:
        return {"ok": False}
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    before_n = len(df_all)
    df_all = df_all[np.isfinite(df_all["B_val"].values)].copy()
    dropped_n = before_n - len(df_all)
    quantiles = np.nanquantile(df_all["B_val"], q=np.linspace(0,1,n_bands+1))
    
    all_snr = df_all["snr_db"].values
    global_bins = np.linspace(np.nanmin(all_snr), np.nanmax(all_snr), bins_snr+1)
    global_centers = 0.5*(global_bins[:-1] + global_bins[1:])
    
    band_means_swer = []
    band_means_per = []
    
    plt.figure(figsize=(6,4))
    for i in range(n_bands):
        qlo, qhi = quantiles[i], quantiles[i+1]
        if i < n_bands - 1:
            band_mask = (df_all["B_val"] >= qlo) & (df_all["B_val"] < qhi)
        else:
            band_mask = (df_all["B_val"] >= qlo) & (df_all["B_val"] <= qhi)
        sub = df_all[band_mask]
        snr = sub["snr_db"].values
        swer = sub["swer"].values
        per = sub["per"].values
        
        idx = np.digitize(snr, global_bins) - 1
        idx = np.clip(idx, 0, bins_snr-1)
        means = np.array([np.nanmean(swer[idx==k]) if np.any(idx==k) else np.nan for k in range(bins_snr)])
        means_per = np.array([np.nanmean(per[idx==k]) if np.any(idx==k) else np.nan for k in range(bins_snr)])
        
        band_means_swer.append(means)
        band_means_per.append(means_per)
        
        plt.plot(global_centers, means, label=f"B band {i+1}")
        
    plt.xlabel("SNR (dB)")
    plt.ylabel("sWER")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "fig02A_snr_swer_bands.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6,4))
    for i in range(n_bands):
        plt.plot(global_centers, band_means_per[i], label=f"B band {i+1}")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "fig02A_snr_per_bands.png"), dpi=200)
    plt.close()
    
    # --- Audit Logic (same SNR window and same gap definition) ---
    tol = 1e-6
    max_gap = 0.0
    max_gap_swer = 0.0
    max_gap_per = 0.0
    n_bins_compared = 0
    per_bin_gap = []
    relevant_indices = np.where((global_centers >= 2.0) & (global_centers <= 20.0))[0]

    for idx in relevant_indices:
        vals_swer = [band_means_swer[b][idx] for b in range(n_bands)]
        vals_per = [band_means_per[b][idx] for b in range(n_bands)]
        v_swer = [v for v in vals_swer if np.isfinite(v)]
        v_per = [v for v in vals_per if np.isfinite(v)]
        gap_swer = (max(v_swer) - min(v_swer)) if len(v_swer) >= 2 else np.nan
        gap_per = (max(v_per) - min(v_per)) if len(v_per) >= 2 else np.nan

        if np.isfinite(gap_swer):
            max_gap_swer = max(max_gap_swer, float(gap_swer))
        if np.isfinite(gap_per):
            max_gap_per = max(max_gap_per, float(gap_per))

        gap_candidates = [g for g in [gap_swer, gap_per] if np.isfinite(g)]
        if gap_candidates:
            bin_gap = float(max(gap_candidates))
            per_bin_gap.append(bin_gap)
            n_bins_compared += 1
            max_gap = max(max_gap, bin_gap)

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

def freq_heatmap_for_algo(data_algo, out_path_base, out_dir, algo_name, window_frac=0.2):
    seeds = sorted(list(data_algo.keys()))
    if len(seeds) == 0:
        return {"ok": False}
    
    all_P = []
    all_B = []
    
    g_ref = 50 # Default, will update
    
    for s in seeds:
        df = data_algo[s]
        g = int(df["grid_density"].iloc[0])
        g_ref = g
        start = int((1.0 - window_frac) * len(df))
        d = df.iloc[start:].copy()
        
        # P: 0..g-1, B: 0..g-1
        # We want to use pcolormesh with correct edges.
        # P index range [0, g-1] maps to [20, 33] dBm
        # B index range [0, g-1] maps to [5, 20] kHz
        
        # We'll just collect raw indices first
        if not d.empty:
            all_P.append(d["P_index"].values)
            all_B.append(d["B_index"].values)

    if not all_P:
        return {"ok": False}

    all_P = np.concatenate(all_P)
    all_B = np.concatenate(all_B)
    
    # Physical ranges
    p_min, p_max = 20.0, 33.0
    b_min, b_max = 5.0, 20.0
    
    # Bin edges for indices [0, g] -> Physical values
    # We want g bins.
    # P_val = p_min + (p_idx / (g-1)) * (p_max - p_min)
    # But for histogram, we want bin edges.
    # Let's define edges in index space first: -0.5 to g-0.5
    idx_edges = np.linspace(-0.5, g_ref-0.5, g_ref+1)
    
    # Map edges to physical space?
    # Actually, let's map the data points to physical space first, then bin.
    # P_mapped = p_min + (P_idx / (g_ref-1)) * (p_max - p_min)
    # This assumes P_idx goes from 0 to g_ref-1 exactly.
    
    # To avoid off-by-one, let's define the physical bins directly.
    # We want exactly g_ref bins covering [p_min, p_max].
    # But wait, the mapping function `map_index_to_value` maps integer indices to specific values.
    # Values are discrete.
    # P_values = [map_index_to_value(i, g_ref, p_min, p_max) for i in range(g_ref)]
    # So we should bin around these discrete values.
    
    # Let's compute counts on the indices [0..g-1] directly, then plot using physical axes.
    H, x_edges_idx, y_edges_idx = np.histogram2d(all_P, all_B, bins=[np.arange(g_ref+1), np.arange(g_ref+1)])
    # H is (g_ref, g_ref). x axis is P, y axis is B.
    # H[i, j] is count for P=i, B=j.
    
    # Normalization: count / total_count
    total_count = np.sum(H)
    H_norm = H / total_count if total_count > 0 else H
    
    # Audit Metrics
    max_cell_share = np.max(H_norm)
    
    flat_p = H_norm.flatten()
    sorted_p = np.sort(flat_p)[::-1] # Descending
    top2_share = np.sum(sorted_p[:2])
    
    # Entropy
    # entropy = -sum(p * log(p))
    eps = 1e-12
    entropy = -np.sum(flat_p * np.log(flat_p + eps))
    
    nonzero_cells = np.sum(H > 0)
    
    # Plotting
    # We need physical edges for pcolormesh
    # P index i corresponds to value v_i.
    # We want the cell for i to be centered at v_i? Or span [v_i - delta, v_i + delta]?
    # Usually pcolormesh edges are boundaries.
    # If we have g_ref bins, we need g_ref+1 edges.
    # P range [20, 33]. g=50.
    # Edges should cover 20 to 33?
    # If P_index=0 -> 20. P_index=49 -> 33.
    # We want the plot to extend from 20 to 33.
    # So edges should be linspace(20, 33, g_ref+1) ? 
    # Let's check: 50 bins. 51 edges.
    # This aligns with P indices 0..49 being mapped to these bins.
    
    p_edges = np.linspace(p_min, p_max, g_ref+1)
    b_edges = np.linspace(b_min, b_max, g_ref+1)
    
    # 1. Linear Plot
    plt.figure(figsize=(5,4))
    # H.T because imshow/pcolormesh expects (y, x) or we carefully specify
    # histogram2d returns H[x, y]. pcolormesh(X, Y, C) expects C[y, x] usually or matches shape.
    # documentation: pcolormesh(X, Y, C). X, Y 1D or 2D. C is (ny-1, nx-1).
    # If X has len nx, Y has len ny.
    # Here X=p_edges (51), Y=b_edges (51). H is (50, 50).
    # So we pass H.T to match (B on y-axis, P on x-axis).
    plt.pcolormesh(p_edges, b_edges, H_norm.T, cmap='viridis', shading='flat')
    plt.colorbar(label="Probability")
    plt.xlabel("Power (dBm)")
    plt.ylabel("Bandwidth (kHz)")
    plt.title(f"{algo_name} (Linear)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_linear_{algo_name}.png"), dpi=200)
    plt.close()
    
    # 2. Log Plot
    # Use vmin=eps (or quantile?), vmax=max
    # User said: "LogNorm(vmin=eps, vmax=max) or quantile clipping"
    # User said: "Single algorithm -> adaptive"
    # We will use LogNorm.
    
    plt.figure(figsize=(5,4))
    plt.pcolormesh(p_edges, b_edges, H_norm.T, cmap='viridis', shading='flat', norm=LogNorm(vmin=1e-6, vmax=max(1e-6, H_norm.max())))
    plt.colorbar(label="Probability (Log)")
    plt.xlabel("Power (dBm)")
    plt.ylabel("Bandwidth (kHz)")
    plt.title(f"{algo_name} (Log)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"heatmap_log_{algo_name}.png"), dpi=200)
    plt.close()
    
    return {
        "ok": True,
        "max_cell_share": float(max_cell_share),
        "top2_share": float(top2_share),
        "entropy": float(entropy),
        "nonzero_cells": int(nonzero_cells)
    }

def fig02B_geometry(data_zoom, out_path, window_frac=0.2):
    res = {}
    for algo in data_zoom:
        res[algo] = freq_heatmap_for_algo(data_zoom[algo], None, out_path, algo, window_frac=window_frac)
    return res

def fig02C_semantic_constraint(data_all, out_path, semantic_col, bins_qsem=6, window_frac=0.2, n_chunks=10):
    # Boxplot logic (Keep it as it provides context, but audit focuses on Pareto)
    data_ra = data_all.get("raucb_plus", {})
    if data_ra:
        seeds = sorted(list(data_ra.keys()))
        dfs = []
        for s in seeds:
             d = data_ra[s]
             if all(c in d.columns for c in ["mos", semantic_col]):
                 dfs.append(d[["mos", semantic_col]])
        
        if dfs:
            d_box = pd.concat(dfs, axis=0, ignore_index=True)
            q = d_box[semantic_col].values
            mos = d_box["mos"].values
            bins = np.linspace(np.nanmin(q), np.nanmax(q), bins_qsem+1)
            centers = 0.5*(bins[:-1] + bins[1:])
            idx = np.digitize(q, bins) - 1
            idx = np.clip(idx, 0, bins_qsem-1)
            vals = [mos[idx==i] for i in range(bins_qsem)]
            plt.figure(figsize=(6,4))
            plt.boxplot([v[~np.isnan(v)] for v in vals], positions=list(range(1,bins_qsem+1)))
            plt.xticks(list(range(1,bins_qsem+1)), [f"{c:.1f}" for c in centers])
            xlabel = semantic_col
            if semantic_col == "q_semantic": xlabel += " (proxy)"
            plt.xlabel(xlabel)
            plt.ylabel("QoE (MOS)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, "fig02C_qoe_vs_qsem_box.png"), dpi=200)
            plt.close()

    # Pareto Logic
    sample_df = None
    for _algo in data_all:
        for _seed, _df in data_all[_algo].items():
            sample_df = _df
            break
        if sample_df is not None:
            break
    if sample_df is None:
        return {"ok": False}
    u_col = "utility_reward" if "utility_reward" in sample_df.columns else "reward"

    algo_points = []
    
    target_algos = ["raucb_plus", "safeopt_gp", "safe_linucb", "sw_ucb", "lagrangian_ppo", "lyapunov_greedy_oracle", "best_fixed_arm_oracle"]
    
    for algo in target_algos:
        if algo not in data_all: continue
        
        for seed, df in data_all[algo].items():
            start = int((1.0 - window_frac) * len(df))
            d_win = df.iloc[start:]
            if "swer" in d_win.columns and u_col in d_win.columns:
                if len(d_win) == 0:
                    continue
                effective_chunks = max(2, int(n_chunks))
                chunk_size = int(math.ceil(len(d_win) / float(effective_chunks)))
                chunk_size = max(1, chunk_size)

                for chunk_id, i in enumerate(range(0, len(d_win), chunk_size)):
                    sub = d_win.iloc[i : i+chunk_size]
                    if sub.empty: continue
                    m_swer = float(np.nanmean(sub["swer"].values))
                    m_u = float(np.nanmean(sub[u_col].values))
                    if np.isfinite(m_swer) and np.isfinite(m_u):
                        algo_points.append({"algo": algo, "seed": int(seed), "chunk_id": int(chunk_id), "swer": m_swer, "U": m_u})

    # Domination Logic
    # b dominates a if b.x <= a.x AND b.y >= a.y (at least one strict)
    # x = swer (minimize), y = U (maximize)
    
    # Filter Non-Dominated Set
    nondominated = []
    for i, a in enumerate(algo_points):
        is_dominated = False
        for j, b in enumerate(algo_points):
            if i == j: continue
            
            # Check if b dominates a
            # b.swer <= a.swer AND b.U >= a.U
            better_eq_swer = b["swer"] <= a["swer"]
            better_eq_u = b["U"] >= a["U"]
            
            if better_eq_swer and better_eq_u:
                # Check for strictness
                strict_swer = b["swer"] < a["swer"]
                strict_u = b["U"] > a["U"]
                if strict_swer or strict_u:
                    is_dominated = True
                    break
        if not is_dominated:
            nondominated.append(a)
            
    # Sort by sWER ascending for plotting line
    nondominated.sort(key=lambda p: p["swer"])

    # explicit per-point certificate
    frontier_checks = []
    for i, a in enumerate(nondominated):
        dominated_by = None
        for j, b in enumerate(algo_points):
            if a is b:
                continue
            better_eq_swer = b["swer"] <= a["swer"]
            better_eq_u = b["U"] >= a["U"]
            strict_swer = b["swer"] < a["swer"]
            strict_u = b["U"] > a["U"]
            if better_eq_swer and better_eq_u and (strict_swer or strict_u):
                dominated_by = {"algo": b["algo"], "seed": b.get("seed"), "chunk_id": b.get("chunk_id")}
                break
        frontier_checks.append({
            "idx": i,
            "algo": a.get("algo"),
            "seed": a.get("seed"),
            "chunk_id": a.get("chunk_id"),
            "pass": dominated_by is None,
            "dominated_by": dominated_by
        })
    
    # Plotting
    plt.figure(figsize=(6,4))
    
    algos_present = sorted(list(set(p["algo"] for p in algo_points)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos_present)))
    c_map = {a: colors[i] for i, a in enumerate(algos_present)}
    
    for algo in algos_present:
        subset = [p for p in algo_points if p["algo"] == algo]
        xs = [p["swer"] for p in subset]
        ys = [p["U"] for p in subset]
        plt.scatter(xs, ys, color=c_map[algo], label=algo, alpha=0.5, s=20)
        
        # Centroid
        if xs:
            mx, my = np.mean(xs), np.mean(ys)
            plt.annotate(algo, (mx, my), fontsize=9, fontweight='bold', color='black')

    # Plot Frontier Line
    fx = [p["swer"] for p in nondominated]
    fy = [p["U"] for p in nondominated]
    plt.plot(fx, fy, 'r--', label='Pareto Frontier', linewidth=2)
    
    plt.xlabel("sWER (lower is better)")
    plt.ylabel(f"U (higher is better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "fig02C_qoe_swer_pareto.png"), dpi=200)
    plt.close()
    
    # Audit: Check if every frontier point is truly non-dominated (self-check)
    # Already done by definition, but we'll report count
    return {
        "ok": True,
        "points_total": len(algo_points),
        "nondominated_count": len(nondominated),
        "frontier_points": nondominated,
        "frontier_checks": frontier_checks,
    }

def generate_audit_report(args, resA, resB, resC):
    lines = []
    # 1. Header (Acceptance Score Standard)
    lines.append("置顶：验收打分标准（必须输出到报告首行）")
    lines.append("Score=10 的条件：")
    lines.append("python -m compileall -q task3/fig02_physics_semantics_geometry.py exit 0")
    lines.append("只做“修改 + 生成 Fig02 审计报告”， 不要跑大规模训练/长实验 （除非已有 CSV）")
    lines.append("审计报告 fig02_audit.md 满足：")
    lines.append("Heatmap mass concentration：最大单元格占比 < 0.60 ；Top-2 合计 < 0.85")
    lines.append("Heatmap entropy（按概率分布计算） > 1.0 （阈值可调，但必须不是接近 0）")
    lines.append("Band curves：PER 或 sWER 三条 band 在 SNR=2~20 的区间内 max_gap > 0.02，且 curves_identical 不得与 max_gap 逻辑矛盾（必须输出 n_bins_compared，>=5 才允许 identical=True）")
    lines.append("Pareto：输出 nondominated 点数 > 3，且每个前沿点都通过“未被支配”检查")
    lines.append("重新生成的 Fig02B 不允许“全黑 + 单点亮”；必须能看出“可行域内形成高密区”的结构趋势【 】")
    lines.append("最后输出 ACCEPTANCE_JSON=... （包含上述指标与 PASS/FAIL）")
    lines.append("")
    lines.append("# Fig02 Audit Report")
    
    pass_all = True
    json_metrics = {}
    
    # 2. Heatmap Audit (Fig02B)
    lines.append("## Fig02B: Heatmap Audit")
    heatmap_metrics = {}
    required_zoom = set(args.algos_zoom)
    for target_algo in sorted(required_zoom):
        rb = resB.get(target_algo)
        if not rb or not rb.get("ok"):
            lines.append(f"### Algorithm: {target_algo}")
            lines.append("- Result: FAIL (missing or invalid heatmap data)")
            pass_all = False
            continue
        lines.append(f"### Algorithm: {target_algo}")
        lines.append(f"- Max Cell Share: {rb['max_cell_share']:.4f} (Limit: < 0.60)")
        lines.append(f"- Top-2 Share: {rb['top2_share']:.4f} (Limit: < 0.85)")
        lines.append(f"- Entropy: {rb['entropy']:.4f} (Limit: > 1.0)")
        lines.append(f"- Nonzero Cells: {rb['nonzero_cells']}")
        cond1 = rb['max_cell_share'] < 0.60
        cond2 = rb['top2_share'] < 0.85
        cond3 = rb['entropy'] > 1.0
        cond4 = rb['nonzero_cells'] > 1
        algo_pass = cond1 and cond2 and cond3 and cond4
        lines.append(f"**Result: {'PASS' if algo_pass else 'FAIL'}**")
        if not algo_pass:
            pass_all = False
        heatmap_metrics[target_algo] = {
            "max_cell_share": rb['max_cell_share'],
            "top2_share": rb['top2_share'],
            "entropy": rb['entropy'],
            "nonzero_cells": rb['nonzero_cells'],
            "pass": algo_pass,
        }
    json_metrics["heatmap_by_algo"] = heatmap_metrics
        
    # 3. Band Curves Audit (Fig02A)
    lines.append("")
    lines.append("## Fig02A: Band Curves Audit")
    if resA.get("ok"):
        max_gap = resA.get("max_gap", 0.0)
        identical = resA.get("curves_identical", False)
        n_bins_compared = int(resA.get("n_bins_compared", 0))
        identical_reason = resA.get("identical_reason", "unknown")
        
        lines.append(f"- Max Gap (SNR 2-20): {max_gap:.4f} (Limit: > 0.02)")
        lines.append(f"- Curves Identical: {identical} (Limit: False)")
        lines.append(f"- n_bins_compared: {n_bins_compared} (Constraint: >=5 only then identical may be True)")
        lines.append(f"- identical_reason: {identical_reason}")
        
        cond_band1 = max_gap > 0.02
        cond_band2 = not identical
        cond_band3 = (n_bins_compared >= 5) or (not identical)
        cond_band4 = not (identical and max_gap > resA.get("tol", 1e-6))
        
        if not (cond_band1 and cond_band2 and cond_band3 and cond_band4):
            pass_all = False
            lines.append("**Result: FAIL**")
        else:
            lines.append("**Result: PASS**")
            
        json_metrics["band_max_gap"] = max_gap
        json_metrics["band_identical"] = identical
        json_metrics["band_n_bins_compared"] = n_bins_compared
        json_metrics["band_identical_reason"] = identical_reason
    else:
        lines.append("Fig02A generation failed.")
        pass_all = False

    # 4. Pareto Audit (Fig02C)
    lines.append("")
    lines.append("## Fig02C: Pareto Audit")
    if resC.get("ok"):
        cnt = resC.get("nondominated_count", 0)
        pts = resC.get("points_total", 0)
        checks = resC.get("frontier_checks", [])
        lines.append(f"- Non-dominated Points: {cnt} (Limit: > 3)")
        lines.append(f"- Total candidate points: {pts}")

        check_pass = True
        lines.append("- Frontier point-by-point non-domination checks:")
        for c in checks:
            ok = bool(c.get("pass", False))
            check_pass = check_pass and ok
            lines.append(
                f"  - idx={c['idx']} algo={c['algo']} seed={c['seed']} chunk_id={c['chunk_id']} => {'PASS' if ok else 'FAIL'}"
            )
        
        cond_pareto = cnt > 3
        if not (cond_pareto and check_pass):
            pass_all = False
            lines.append("**Result: FAIL**")
        else:
            lines.append("**Result: PASS**")
            
        json_metrics["pareto_points"] = cnt
        json_metrics["pareto_points_total"] = pts
        json_metrics["pareto_frontier_checks_pass"] = check_pass
    else:
        lines.append("Fig02C generation failed.")
        pass_all = False
        
    lines.append("")
    json_metrics["overall_pass"] = pass_all
    
    # Write File
    with open(os.path.join(args.out_dir, "fig02_audit.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    # Return JSON string
    return json.dumps(json_metrics)

def main():
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    data_all = load_csvs(args.input_dir, args.T, None)
    if len(data_all) == 0:
        print("[FATAL] No CSVs loaded")
        sys.exit(2)
        
    # Determine semantic col
    semantic_col = None
    search_algos = ["raucb_plus"] + sorted(list(data_all.keys()))
    for algo in search_algos:
        if algo in data_all:
            first_seed = list(data_all[algo].keys())[0]
            df_sample = data_all[algo][first_seed]
            if "s_t" in df_sample.columns: 
                semantic_col = "s_t"
                break
            if "semantic_weight" in df_sample.columns: 
                semantic_col = "semantic_weight"
                break
    if semantic_col is None and args.allow_proxy:
        semantic_col = "q_semantic"

    # Fig02A
    resA = fig02A_physics(data_all, args.out_dir, bins_snr=args.bins_snr)
    # Check bands if columns exist
    do_bands = True
    for algo in data_all:
        for seed, df in data_all[algo].items():
            if "B_index" not in df.columns or "grid_density" not in df.columns:
                do_bands = False
    
    if resA.get("ok") and do_bands:
        res_bands = fig02A_optional_B_bands(data_all, args.out_dir, bins_snr=args.bins_snr)
        resA.update(res_bands) # Merge band results into resA

    # Fig02B
    data_zoom = load_csvs(args.input_dir, args.T, args.algos_zoom)
    resB = fig02B_geometry(data_zoom, args.out_dir, window_frac=args.window_frac)

    # Fig02C
    resC = {"ok": False}
    if semantic_col:
        resC = fig02C_semantic_constraint(data_all, args.out_dir, semantic_col, bins_qsem=args.bins_qsem, window_frac=args.window_frac, n_chunks=args.pareto_chunks)

    # Audit Report
    acc_json = generate_audit_report(args, resA, resB, resC)
    print(f"ACCEPTANCE_JSON={acc_json}")

if __name__ == "__main__":
    main()
