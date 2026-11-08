IMU_COLS = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]


def preprocess_imu(df, demo):
    """Preprocess IMU data (acc_*, rot_*).

    Input:
        df   : raw sensor data (train or test), one row per time step
        demo : demographics table, one row per subject
    Output:
        DataFrame with the same number of rows as df, containing
        sequence_id, sequence_counter and the IMU columns.
    """
    out = df[["sequence_id", "sequence_counter"] + IMU_COLS].copy()

    # TODO (Dung): add IMU preprocessing steps here, for example:
    # - handle missing quaternion (rot_*) values
    # - mirror data of left-handed subjects (use demo["handedness"])
    # - add new features (acc magnitude, jerk, ...)
    # Each step must work on one sequence at a time and also on the test set.

    return out
