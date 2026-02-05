import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_clean import FILE_PATH, clean_and_enhance_data, filter_to_lck, create_games_df

def calculate_net_features(df_lck, df_t_all):
    """
    Creates 'Net' versions of player stats by subtracting 1/5th of the team's total lead.
    This isolates individual performance from team-wide momentum.
    """
    # Features where 'Net' adjustment is most vital
    base_features = ['golddiffat15', 'csdiffat15', 'xpdiffat15', 'golddiffat25', 'csdiffat25', 'xpdiffat25']
    
    # 1. Create a lookup for Team Gold/XP/CS differences at specific time marks
    # Team rows in OE data have 'position' == 'team'
    team_stats = df_t_all[df_t_all['position'] == 'team'][['gameid', 'teamname'] + base_features].copy()
    team_stats.columns = ['gameid', 'teamname'] + [f'team_{f}' for f in base_features]
    
    # 2. Merge Team stats into Player stats
    df_net = pd.merge(df_lck, team_stats, on=['gameid', 'teamname'], how='left')
    
    # 3. Calculate Net Delta: Player_Stat - (Team_Stat / 5)
    for f in base_features:
        df_net[f'net_{f}'] = df_net[f] - (df_net[f'team_{f}'] / 5)
        
    return df_net

def optimize_libr_weights_net(df_net):
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    
    # We use 'Net' versions of the deltas, and standard versions of internal shares
    features = [
        'net_golddiffat15', 'net_csdiffat15', 'net_xpdiffat15', 
        'net_golddiffat25', 'net_csdiffat25', 'net_xpdiffat25', 
        'damageshare', 'earnedgoldshare', 'kp',
        'dpm', 'earned gpm', 'cspm', 'vspm', 'wpm', 
        'kills', 'deaths', 'assists'
    ]
    
    optimized_weights = {}

    for role in roles:
        role_df = df_net[df_net['position'] == role].dropna(subset=features + ['result'])
        
        X = role_df[features]
        y = role_df['result']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Logistic Regression handles the 'Importance' calculation
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        
        importance = np.abs(model.coef_[0])
        normalized_weights = (importance / importance.sum()) * 100
        
        optimized_weights[role] = dict(zip(features, np.round(normalized_weights, 2)))

    return optimized_weights

if __name__ == "__main__":
    # Load Data
    df_p, df_t = clean_and_enhance_data(FILE_PATH)
    df_lck, df_t_all, _ = filter_to_lck(df_p, df_t, create_games_df(df_t))
    
    # Step 1: Calculate Net Contribution Stats
    print("Calculating Net Contribution features...")
    df_net = calculate_net_features(df_lck, df_t_all)
    
    # Step 2: Run Optimization on the Adjusted Data
    print("Running Mathematical Weight Optimization on Net Stats...")
    final_weights = optimize_libr_weights_net(df_net)
    
    # Print results for the Carries
    for role in ['top', 'mid', 'bot']:
        print(f"\n--- Optimized {role.upper()} Net Weights ---")
        sorted_w = sorted(final_weights[role].items(), key=lambda x: x[1], reverse=True)
        for feat, weight in sorted_w[:8]:
            print(f"{feat:<20}: {weight}%")