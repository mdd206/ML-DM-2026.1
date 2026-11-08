def preprocess_tof(df, demo):
    """Preprocess ToF (tof_*) and thermal (thm_*) data.

    Input:
        df   : raw sensor data (train or test), one row per time step
        demo : demographics table, one row per subject (not used yet)
    Output:
        DataFrame with the same number of rows as df, containing
        sequence_id, sequence_counter and the tof_* and thm_* columns.
    """
    tof_thm_cols = [c for c in df.columns if c.startswith(("tof_", "thm_"))]
    out = df[["sequence_id", "sequence_counter"] + tof_thm_cols].copy()

    # TODO (Huy): add ToF / thermal preprocessing steps here, for example:
    # - treat -1 ("no object detected") differently from NaN ("not recorded")
    # - add flag columns for broken sensors
    # - summarize each 8x8 ToF grid (mean, std, active pixels, ...)
    # Each step must work on one sequence at a time and also on the test set.

    return out
