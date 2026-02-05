import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from data_clean import FILE_PATH, clean_and_enhance_data, filter_to_lck, create_games_df

DIAG_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diagnostics"))

def plot_role_linearity(df, feature, role, bins=8):
    role_df = df[df['position'] == role].copy()
    if len(role_df) < 20: return
    
    role_dir = os.path.join(DIAG_BASE_DIR, role)
    if not os.path.exists(role_dir): os.makedirs(role_dir)

    try:
        role_df['bin'] = pd.qcut(role_df[feature], bins, duplicates='drop')
    except:
        role_df['bin'] = pd.cut(role_df[feature], bins)

    stats = role_df.groupby('bin', observed=True).agg({'result': 'mean', feature: 'mean'})
    stats.columns = ['win_rate', 'feature_mean']
    
    eps = 1e-5
    stats['log_odds'] = np.log((stats['win_rate'] + eps) / (1 - stats['win_rate'] + eps))
    
    plt.figure(figsize=(8, 5))
    plt.plot(stats['feature_mean'], stats['log_odds'], marker='o', color='darkorange', label='Observed Log-Odds')
    
    z = np.polyfit(stats['feature_mean'], stats['log_odds'], 1)
    p = np.poly1d(z)
    plt.plot(stats['feature_mean'], p(stats['feature_mean']), "k--", alpha=0.3, label='Linear Fit')
    
    plt.title(f"{role.upper()} | {feature} vs Win Log-Odds")
    plt.xlabel(f"Mean {feature}")
    plt.ylabel("Log-Odds of Winning")
    plt.legend()
    
    plt.savefig(os.path.join(role_dir, f"{feature.lower()}.png"))
    plt.close()

def find_high_correlations(df, features, threshold=0.7):
    """
    Identifies pairs of features with a correlation higher than the threshold.
    Returns a list of dictionaries for easy reading or conversion to a DataFrame.
    """
    # 1. Calculate the correlation matrix
    corr_matrix = df[features].corr().abs()
    
    # 2. Select the upper triangle of the matrix to avoid duplicates (A vs B and B vs A)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 3. Find index/column pairs where correlation > threshold
    high_corr_pairs = []
    for column in upper.columns:
        for index in upper.index:
            val = upper.loc[index, column]
            if val > threshold:
                high_corr_pairs.append({
                    'Feature 1': index,
                    'Feature 2': column,
                    'Correlation': round(val, 3)
                })
    
    return high_corr_pairs

def plot_correlation_heatmap(df, role, features):
    """Checks for redundancy (Multicollinearity) between Raw and Share stats."""
    role_dir = os.path.join(DIAG_BASE_DIR, role)
    if not os.path.exists(role_dir): 
        os.makedirs(role_dir)
        print(f"Created directory: {role_dir}")
    role_df = df[df['position'] == role][features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(role_df, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"{role.upper()} Feature Correlation (Raw vs Share)")
    
    role_dir = os.path.join(DIAG_BASE_DIR, role)
    plt.savefig(os.path.join(role_dir, "feature_correlation.png"))
    plt.close()

if __name__ == "__main__":
    df_p, df_t = clean_and_enhance_data(FILE_PATH)
    df_lck, _, _ = filter_to_lck(df_p, df_t, create_games_df(df_t))

    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    
    # Testing Raw vs Share metrics
    features_to_test = [
        # Relative/Share Stats
        'golddiffat15', 'csdiffat15', 'xpdiffat15', 'golddiffat25', 'csdiffat25', 'xpdiffat25', "damageshare", "earnedgoldshare", 'kp',
        # Raw/Scale Stats
        'dpm', 'earned gpm', 'cspm', 'vspm', 'wpm', "kills", "deaths", "assists", "earnedgold"
    ]

    for role in roles:
        print(f"Analyzing {role.upper()}...")
        plot_correlation_heatmap(df_lck, role, features_to_test)
        for feat in features_to_test:
            if feat in df_lck.columns:
                plot_role_linearity(df_lck, feat, role)

        print(f"\n[ {role.upper()} ] High Correlation Pairs (>0.7):")
        role_df = df_lck[df_lck['position'] == role]
        pairs = find_high_correlations(role_df, features_to_test, threshold=0.7)
        
        if not pairs:
            print("  None found.")
        else:
            # Print as a nice table
            print(pd.DataFrame(pairs).to_string(index=False))

    print(f"\nDiagnostics complete. Check /diagnostics for the Raw vs Share comparisons.")