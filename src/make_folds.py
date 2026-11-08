import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# 1. Read only the needed columns (train.csv is very large)
train = pd.read_csv("data/raw/train.csv",
                    usecols=["sequence_id", "subject", "gesture"])

# 2. Keep one row per sequence (all rows of a sequence share subject and gesture)
seq = train.drop_duplicates("sequence_id").reset_index(drop=True)

# 3. Split into 5 folds: group by subject, stratify by gesture
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
seq["fold"] = -1
splits = sgkf.split(seq, y=seq["gesture"], groups=seq["subject"])
for fold, (train_idx, val_idx) in enumerate(splits):
    seq.loc[val_idx, "fold"] = fold

# 4. Save the result
seq.to_csv("folds/folds.csv", index=False)
print("Saved folds/folds.csv:", seq.shape)
