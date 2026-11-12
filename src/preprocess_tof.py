import numpy as np
import pandas as pd

THM_COLS = [f"thm_{i}" for i in range(1, 6)]
TOF_SENSOR_COLS = {
    f"tof_{i}": [f"tof_{i}_v{p}" for p in range(64)] for i in range(1, 6)
}
TOF_COLS = [col for i in range(1, 6) for col in TOF_SENSOR_COLS[f"tof_{i}"]]

# Physical ceiling (mm) for "no object detected" (-1), just above max valid reading (249 mm)
NO_OBJECT_DIST = 255.0
# Minimum normal wrist/ambient temperature (°C); below this indicates sensor offset or dead channel
NORMAL_THM_FLOOR = 20.0


def preprocess_tof(df, demo):
    """Preprocess ToF (tof_*) and thermal (thm_*) data.

    Input:
        df   : raw sensor data (train or test), one row per time step
        demo : demographics table, one row per subject (not needed for ToF/THM)
    Output:
        DataFrame with the same number of rows as df, containing
        sequence_id, sequence_counter and the processed tof_* and thm_* columns.
    """
    tof_thm_cols = [c for c in df.columns if c.startswith(("tof_", "thm_"))]
    out = df[["sequence_id", "sequence_counter"] + tof_thm_cols].copy()

    if len(out) == 0:
        return out

    # Ensure rows are ordered by time step within each sequence while keeping original index
    out = out.sort_values(["sequence_id", "sequence_counter"], kind="mergesort")
    new_features = {}

    # =========================================================================
    # 1. THERMAL SENSORS (thm_1 .. thm_5)
    # =========================================================================
    seq_grp = out.groupby("sequence_id", sort=False)

    # Detect dead flatline sensors per sequence (e.g. SUBJ_028998 thm_3 == 0.0 throughout)
    # Active offset sensors (e.g. SUBJ_036405 thm_1 with std > 0.2) are kept to preserve delta_thm
    seq_std = seq_grp[THM_COLS].transform("std").fillna(0.0)
    seq_mean = seq_grp[THM_COLS].transform("mean")
    dead_flatline = (seq_mean < NORMAL_THM_FLOOR) & (seq_std < 1e-4)

    for c in THM_COLS:
        is_broken = out[c].isna() | dead_flatline[c]
        new_features[f"{c}_broken"] = is_broken.astype(np.int8)
        out.loc[dead_flatline[c], c] = np.nan

    # Interpolate partial missing values strictly within each sequence
    out[THM_COLS] = out.groupby("sequence_id", sort=False)[THM_COLS].ffill()
    out[THM_COLS] = out.groupby("sequence_id", sort=False)[THM_COLS].bfill()

    # Compute start-of-sequence baseline (mean of first 5 steps: sequence_counter < 5)
    is_start = out["sequence_counter"] < 5
    start_df = out[THM_COLS].where(is_start)
    seq_base = start_df.groupby(out["sequence_id"], sort=False).transform("mean")
    seq_base = seq_base.fillna(out.groupby("sequence_id", sort=False)[THM_COLS].transform("first"))

    # Relative temperature change cancels subject baseline shifts and calibration offsets
    delta_df = out[THM_COLS] - seq_base

    # If a channel is broken for the entire sequence, fill its delta with the row mean of working channels
    row_mean_delta = delta_df.mean(axis=1).fillna(0.0)
    for c in THM_COLS:
        delta_df[c] = delta_df[c].fillna(row_mean_delta)
        new_features[f"{c}_delta"] = delta_df[c].to_numpy(dtype=np.float32)

    # Align absolute temperature for broken or offset-shifted (< 20°C) channels using healthy sensors
    healthy_base = seq_base.where(seq_base >= NORMAL_THM_FLOOR).mean(axis=1).fillna(27.0)
    for c in THM_COLS:
        needs_align = out[c].isna() | (seq_base[c] < NORMAL_THM_FLOOR)
        out.loc[needs_align, c] = healthy_base[needs_align] + delta_df.loc[needs_align, c]

    # Cross-sensor thermal summaries and spatial contrast (upper/side thm_1..3 vs lower thm_4..5)
    delta_arr = delta_df.to_numpy(dtype=np.float32)
    new_features["thm_delta_mean"] = delta_arr.mean(axis=1)
    new_features["thm_delta_std"] = delta_arr.std(axis=1)
    new_features["thm_delta_max"] = delta_arr.max(axis=1)
    new_features["thm_delta_min"] = delta_arr.min(axis=1)
    new_features["thm_123_vs_45_delta"] = (
        delta_arr[:, :3].mean(axis=1) - delta_arr[:, 3:].mean(axis=1)
    )

    thm_mean_series = pd.Series(new_features["thm_delta_mean"], index=out.index)
    new_features["thm_delta_diff"] = (
        thm_mean_series.groupby(out["sequence_id"], sort=False)
        .diff()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    # =========================================================================
    # 2. TIME-OF-FLIGHT SENSORS (tof_1 .. tof_5, 8x8 grids)
    # =========================================================================
    n_rows = len(out)
    stat_names = ["active", "mean", "std", "min", "max", "lr_diff", "tb_diff"]
    tof_stats = {s: np.empty((n_rows, 5), dtype=np.float32) for s in stat_names}

    for idx, i in enumerate(range(1, 6)):
        cols = TOF_SENSOR_COLS[f"tof_{i}"]
        arr = out[cols].to_numpy(dtype=np.float32)

        is_nan = np.isnan(arr)
        valid_mask = (~is_nan) & (arr != -1.0)

        n_nan = is_nan.sum(axis=1)
        n_valid = valid_mask.sum(axis=1)

        broken = n_nan == 64                     # Sensor completely missing (all NaN)
        no_obj = (~broken) & (n_valid == 0)      # Working sensor, no object in range (all -1)

        new_features[f"tof_{i}_broken"] = broken.astype(np.int8)

        # Active pixel fraction (0..1)
        active = n_valid.astype(np.float32) / 64.0
        active[broken] = np.nan

        # Distance statistics over valid pixels (excluding -1 and NaN)
        safe_count = np.maximum(n_valid, 1).astype(np.float32)
        valid_vals = np.where(valid_mask, arr, 0.0)

        mean_val = valid_vals.sum(axis=1) / safe_count
        sq_diff = np.where(valid_mask, (arr - mean_val[:, None]) ** 2, 0.0)
        std_val = np.sqrt(sq_diff.sum(axis=1) / safe_count)
        min_val = np.where(valid_mask, arr, np.inf).min(axis=1)
        max_val = np.where(valid_mask, arr, -np.inf).max(axis=1)

        # When no object is detected (all -1), set distance to physical ceiling (255.0 mm)
        mean_val[no_obj] = NO_OBJECT_DIST
        min_val[no_obj] = NO_OBJECT_DIST
        max_val[no_obj] = NO_OBJECT_DIST
        std_val[no_obj] = 0.0

        for s_arr in (mean_val, std_val, min_val, max_val):
            s_arr[broken] = np.nan

        # 4x4 Quadrant directional asymmetries on the 8x8 grid
        grid_valid = valid_mask.reshape(n_rows, 8, 8).astype(np.float32)
        q1 = grid_valid[:, :4, :4].mean(axis=(1, 2))  # Top-Left
        q2 = grid_valid[:, :4, 4:].mean(axis=(1, 2))  # Top-Right
        q3 = grid_valid[:, 4:, :4].mean(axis=(1, 2))  # Bottom-Left
        q4 = grid_valid[:, 4:, 4:].mean(axis=(1, 2))  # Bottom-Right

        lr_diff = 0.5 * (q2 + q4) - 0.5 * (q1 + q3)
        tb_diff = 0.5 * (q1 + q2) - 0.5 * (q3 + q4)
        lr_diff[broken] = np.nan
        tb_diff[broken] = np.nan

        tof_stats["active"][:, idx] = active
        tof_stats["mean"][:, idx] = mean_val
        tof_stats["std"][:, idx] = std_val
        tof_stats["min"][:, idx] = min_val
        tof_stats["max"][:, idx] = max_val
        tof_stats["lr_diff"][:, idx] = lr_diff
        tof_stats["tb_diff"][:, idx] = tb_diff

        # Clean raw pixel columns: map -1 ("no object") to 255.0 mm
        out[cols] = np.where(arr == -1.0, NO_OBJECT_DIST, arr)

    # Impute raw ToF pixels per sequence for broken steps, fallback to 255.0 mm
    out[TOF_COLS] = out.groupby("sequence_id", sort=False)[TOF_COLS].ffill()
    out[TOF_COLS] = out.groupby("sequence_id", sort=False)[TOF_COLS].bfill()
    out[TOF_COLS] = out[TOF_COLS].fillna(NO_OBJECT_DIST)

    # Impute per-sensor ToF summary features within each sequence, then cross-sensor fallback
    defaults = {
        "active": 0.0,
        "mean": NO_OBJECT_DIST,
        "std": 0.0,
        "min": NO_OBJECT_DIST,
        "max": NO_OBJECT_DIST,
        "lr_diff": 0.0,
        "tb_diff": 0.0,
    }
    for s_name in stat_names:
        s_cols = [f"tof_{i}_{s_name}" for i in range(1, 6)]
        s_df = pd.DataFrame(tof_stats[s_name], index=out.index, columns=s_cols)
        s_df = s_df.groupby(out["sequence_id"], sort=False).ffill()
        s_df = s_df.groupby(out["sequence_id"], sort=False).bfill()

        row_avg = s_df.mean(axis=1).fillna(defaults[s_name])
        for i, c in enumerate(s_cols):
            s_df[c] = s_df[c].fillna(row_avg)
            new_features[c] = s_df[c].to_numpy(dtype=np.float32)
        tof_stats[s_name] = s_df.to_numpy(dtype=np.float32)

    # Cross-sensor ToF aggregations (separates BFRB face/neck from knee/leg touches)
    act_mat = tof_stats["active"]
    mean_mat = tof_stats["mean"]
    min_mat = tof_stats["min"]

    new_features["tof_active_mean"] = act_mat.mean(axis=1)
    new_features["tof_active_std"] = act_mat.std(axis=1)
    new_features["tof_mean_dist_all"] = mean_mat.mean(axis=1)
    new_features["tof_min_dist_all"] = min_mat.min(axis=1)
    new_features["tof_5_vs_1234_active"] = act_mat[:, 4] - act_mat[:, :4].mean(axis=1)
    new_features["tof_5_vs_3_mean"] = mean_mat[:, 4] - mean_mat[:, 2]

    min_dist_series = pd.Series(new_features["tof_min_dist_all"], index=out.index)
    new_features["tof_min_dist_diff"] = (
        min_dist_series.groupby(out["sequence_id"], sort=False)
        .diff()
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    # Attach all engineered features at once and restore original row order
    feat_df = pd.DataFrame(new_features, index=out.index)
    out = pd.concat([out, feat_df], axis=1)
    return out.sort_index()