import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def prepare_stratified_folds(df, target_col='Compound', n_splits=5, seed=42):
    """Split instances into n folds grouped by compound, stratified by mean score."""
    df = df.copy()

    # 1. Mean score per compound (target level)
    df['mean_score'] = (df['ModAvg'] + df['HeadAvg']) / 2
    target_stats = df.groupby(target_col)['mean_score'].mean().reset_index()

    # 2. Binning with fallback if the split edges collide
    try:
        target_stats['bin'] = pd.qcut(
            target_stats['mean_score'], q=n_splits, labels=False, duplicates='drop'
        )
    except ValueError:
        target_stats['bin'] = pd.cut(
            target_stats['mean_score'], bins=n_splits, labels=False
        )

    # 3. Stratified grouping folds (compound never leaks across folds)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    target_folds = {}
    for fold, (_, val_idx) in enumerate(
        sgkf.split(X=target_stats, y=target_stats['bin'], groups=target_stats[target_col])
    ):
        for idx in val_idx:
            compound_val = target_stats[target_col].iloc[idx]
            target_folds[compound_val] = fold

    # 4. Map fold back onto the original rows
    df['fold'] = df[target_col].map(target_folds)
    df.drop(columns=['mean_score'], inplace=True)

    fold_counts = df['fold'].value_counts().to_dict()
    print(f'Prepared {n_splits} folds. Records per fold: {fold_counts}')

    return df