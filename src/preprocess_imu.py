import numpy as np

ACC_COLS = ["acc_x", "acc_y", "acc_z"]
ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]
IMU_COLS = ACC_COLS + ROT_COLS

FEATURE_COLS = [
    "rot_missing",
    "acc_mag",
    "acc_pitch",
    "acc_roll",
    "acc_jerk",
    "acc_mag_roll_mean_5",
    "acc_mag_roll_std_5",
    "acc_dyn_mag",
    "rot_roll",
    "rot_pitch",
    "rot_yaw",
    "rot_ang_vel",
]


def _handle_missing_values(work):
    """Fill missing IMU values per sequence and normalize quaternions."""
    work["rot_missing"] = work[ROT_COLS].isna().any(axis=1).astype(np.int8)

    # Impute strictly within each sequence (separate groupby calls so bfill never crosses sequences)
    work[IMU_COLS] = work.groupby("sequence_id", sort=False)[IMU_COLS].ffill()
    work[IMU_COLS] = work.groupby("sequence_id", sort=False)[IMU_COLS].bfill()
    work[ACC_COLS] = work[ACC_COLS].fillna(0.0)

    # Fill sequences missing orientation entirely with the identity quaternion (1, 0, 0, 0)
    still_missing = work[ROT_COLS].isna().any(axis=1)
    if still_missing.any():
        work.loc[still_missing, "rot_w"] = 1.0
        work.loc[still_missing, ["rot_x", "rot_y", "rot_z"]] = 0.0

    # Normalize to unit quaternions; fall back to identity if norm is degenerate
    q = work[ROT_COLS].to_numpy(dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-8
    q[valid] = q[valid] / norm[valid]
    q[~valid] = [1.0, 0.0, 0.0, 0.0]
    work[ROT_COLS] = q


def _mirror_left_handed(work, demo):
    """Mirror IMU axes across the YZ plane (x -> -x) for left-handed subjects."""
    if demo is None or "_subject" not in work.columns or "handedness" not in demo.columns:
        return

    # handedness: 0 = left-handed, 1 = right-handed
    handedness_map = demo.drop_duplicates("subject").set_index("subject")["handedness"]
    handedness = work["_subject"].map(handedness_map).fillna(1)
    left_mask = (handedness == 0).to_numpy()

    if not left_mask.any():
        return

    work.loc[left_mask, "acc_x"] = -work.loc[left_mask, "acc_x"]

    # Reflection M = diag(-1, 1, 1) conjugates rotation matrix R -> M R M,
    # mapping unit quaternion (w, x, y, z) to (w, x, -y, -z)
    rot_mask = left_mask & (work["rot_missing"].to_numpy() == 0)
    work.loc[rot_mask, ["rot_y", "rot_z"]] = -work.loc[rot_mask, ["rot_y", "rot_z"]]


def _add_acc_features(work):
    """Add magnitude, wrist tilt angles, jerk, and local rolling acceleration features."""
    ax = work["acc_x"].to_numpy(dtype=np.float64)
    ay = work["acc_y"].to_numpy(dtype=np.float64)
    az = work["acc_z"].to_numpy(dtype=np.float64)

    work["acc_mag"] = np.sqrt(ax * ax + ay * ay + az * az)
    # Wrist tilt angles capture gravity direction (key for posture separation)
    work["acc_pitch"] = np.arctan2(ay, np.sqrt(ax * ax + az * az))
    work["acc_roll"] = np.arctan2(-ax, az)

    seq_group = work.groupby("sequence_id", sort=False)

    # Step-to-step acceleration change (jerk) within each sequence
    d_acc = seq_group[ACC_COLS].diff().fillna(0.0).to_numpy(dtype=np.float64)
    work["acc_jerk"] = np.linalg.norm(d_acc, axis=1)

    # Short centered rolling window captures local motion intensity and tremor amplitude
    roll_5 = seq_group["acc_mag"].rolling(window=5, min_periods=1, center=True)
    work["acc_mag_roll_mean_5"] = roll_5.mean().reset_index(level=0, drop=True)
    work["acc_mag_roll_std_5"] = roll_5.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)

    # Subtract wider rolling mean (gravity baseline) to isolate dynamic hand movement
    grav_est = (
        seq_group[ACC_COLS]
        .rolling(window=11, min_periods=1, center=True)
        .mean()
        .reset_index(level=0, drop=True)
        .to_numpy(dtype=np.float64)
    )
    dyn_acc = work[ACC_COLS].to_numpy(dtype=np.float64) - grav_est
    work["acc_dyn_mag"] = np.linalg.norm(dyn_acc, axis=1)


def _add_rot_features(work):
    """Add Euler angles and step-to-step angular velocity from unit quaternions."""
    w = work["rot_w"].to_numpy(dtype=np.float64)
    x = work["rot_x"].to_numpy(dtype=np.float64)
    y = work["rot_y"].to_numpy(dtype=np.float64)
    z = work["rot_z"].to_numpy(dtype=np.float64)

    # Intrinsic ZYX / extrinsic XYZ Euler angles (radians)
    work["rot_roll"] = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    work["rot_pitch"] = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    work["rot_yaw"] = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Angular displacement between consecutive steps using |q_t . q_{t-1}|
    # (absolute dot product avoids pi jumps caused by q == -q sign ambiguity)
    q_curr = work[ROT_COLS].to_numpy(dtype=np.float64)
    q_prev = work.groupby("sequence_id", sort=False)[ROT_COLS].shift(1)
    q_prev = q_prev.fillna(work[ROT_COLS]).to_numpy(dtype=np.float64)
    dot = np.clip(np.abs(np.sum(q_curr * q_prev, axis=1)), 0.0, 1.0)
    work["rot_ang_vel"] = 2.0 * np.arccos(dot)


def preprocess_imu(df, demo):
    """Preprocess IMU data (acc_*, rot_*).

    Input:
        df   : raw sensor data (train or test), one row per time step
        demo : demographics table, one row per subject
    Output:
        DataFrame with the same number of rows as df, containing
        sequence_id, sequence_counter and the IMU columns.
    """
    out_cols = ["sequence_id", "sequence_counter"] + IMU_COLS + FEATURE_COLS
    if len(df) == 0:
        return df[["sequence_id", "sequence_counter"] + IMU_COLS].reindex(columns=out_cols)

    # Sort inside each sequence for temporal features while preserving original row order
    work = df[["sequence_id", "sequence_counter"] + IMU_COLS].copy()
    if "subject" in df.columns:
        work["_subject"] = df["subject"].to_numpy()
    work["_orig_pos"] = np.arange(len(df))
    work = work.sort_values(["sequence_id", "sequence_counter"], kind="mergesort").reset_index(
        drop=True
    )

    _handle_missing_values(work)
    _mirror_left_handed(work, demo)
    _add_acc_features(work)
    _add_rot_features(work)

    work = work.sort_values("_orig_pos", kind="mergesort").reset_index(drop=True)
    return work[out_cols]
