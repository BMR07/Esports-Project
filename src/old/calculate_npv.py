import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from data_clean import FILE_PATH, clean_and_enhance_data, filter_to_lck, create_games_df
from optimize_params import calculate_net_features

def get_trained_models(df_net):
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    features = [
        'net_golddiffat15', 'net_csdiffat15', 'net_xpdiffat15', 
        'net_golddiffat25', 'net_csdiffat25', 'net_xpdiffat25', 
        'damageshare', 'earnedgoldshare', 'kp',
        'dpm', 'earned gpm', 'cspm', 'vspm', 'wpm', 
        'kills', 'deaths', 'assists'
    ]
    
    models = {}
    scalers = {}

    for role in roles:
        role_df = df_net[df_net['position'] == role].dropna(subset=features + [ 'result'])
        X = role_df[features]
        y = role_df['result']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        
        models[role] = model
        scalers[role] = scaler
        
    return models, scalers, features

def calculate_zero_sum_npv(df_net, models, scalers, features):
    """
    Calculates Wins Contributed by distributing the actual match result (1 or 0)
    across teammates based on their relative performance strength.
    """
    results = []
    # 1. Calculate raw P(Win) for every player
    for role, model in models.items():
        role_df = df_net[df_net['position'] == role].copy()
        X_scaled = scalers[role].transform(role_df[features].fillna(0))
        role_df['p_win_raw'] = model.predict_proba(X_scaled)[:, 1]
        results.append(role_df)
    
    df_combined = pd.concat(results)

    # 2. Calculate the "Performance Share" within the team for each game
    # This prevents the 'GenG Inflation' by ensuring the team only shares 1 win.
    team_total_p = df_combined.groupby(['gameid', 'teamname'])['p_win_raw'].transform('sum')
    df_combined['contribution_share'] = df_combined['p_win_raw'] / team_total_p

    # 3. Calculate Wins Contributed
    # (Result - 0.5) centers the game around 0. 
    # Win = +0.5, Loss = -0.5. 
    # We multiply by 2 so a 'Perfect' team win equals 1.0 total wins contributed.
    df_combined['wins_contributed'] = (df_combined['result'] - 0.5) * 2 * df_combined['contribution_share']
    
    return df_combined

if __name__ == "__main__":
    # Load and Prepare
    df_p, df_t = clean_and_enhance_data(FILE_PATH)
    df_lck, df_t_all, _ = filter_to_lck(df_p, df_t, create_games_df(df_t))
    df_net = calculate_net_features(df_lck, df_t_all)
    
    # Get Models
    print("Training predictive models per role...")
    models, scalers, features = get_trained_models(df_net)
    
    # Calculate Zero-Sum NPV
    print("Calculating Zero-Sum Wins Contributed...")
    df_final = calculate_zero_sum_npv(df_net, models, scalers, features)
    
    # Leaderboard: Cumulative Wins Contributed
    leaderboard = df_final.groupby(['playername', 'position']).agg({
        'wins_contributed': 'sum',
        'result': 'count'
    }).rename(columns={'result': 'games'}).sort_values('wins_contributed', ascending=False)
    
    print("\n--- 2023 LCK Wins Contributed (Zero-Sum NPV) ---")
    # This leaderboard will show how many 'Actual Wins' the player was responsible for.
    print(leaderboard.head(20))