import pandas as pd
import numpy as np

def calculate_npv(df_players, models, scalers, features, agency_map=None):
    """
    Calculates Wins Contributed.
    
    Modes:
    1. Raw Mode (agency_map=None): Distributes credit strictly by P(Win) strength.
    2. Weighted Mode (agency_map={...}): Biases credit toward high-agency roles.
    """
    results = []
    
    # 1. Generate Raw Win Probabilities
    for role, model in models.items():
        role_df = df_players[df_players['position'] == role].copy()
        if role_df.empty: continue
        
        # Scale and Predict
        X_scaled = scalers[role].transform(role_df[features].fillna(0))
        role_df['p_win_raw'] = model.predict_proba(X_scaled)[:, 1]
        results.append(role_df)
    
    df_scored = pd.concat(results)
    
    # 2. Determine Contribution Logic
    if agency_map:
        # MAP the agency weights to the rows
        # We ensure weights are positive floats
        df_scored['agency_weight'] = df_scored['position'].map(agency_map).astype(float)
        
        # Weighted P(Win): High Agency roles get their P(Win) magnified
        df_scored['weighted_p_win'] = df_scored['p_win_raw'] * df_scored['agency_weight']
        
        # Calculate Share based on WEIGHTED totals
        team_total = df_scored.groupby(['gameid', 'teamname'])['weighted_p_win'].transform('sum')
        df_scored['contribution_share'] = df_scored['weighted_p_win'] / team_total
        
    else:
        # Standard Raw Share (Pass 1)
        team_total = df_scored.groupby(['gameid', 'teamname'])['p_win_raw'].transform('sum')
        df_scored['contribution_share'] = df_scored['p_win_raw'] / team_total

    # 3. Calculate Final NPV (Wins Contributed)
    # (Result - 0.5) * 2 transforms: Win(1)->1, Loss(0)->-1
    # Distributes the single team Win/Loss unit according to share
    df_scored['npv_wins'] = (df_scored['result'] - 0.5) * 2 * df_scored['contribution_share']
    
    return df_scored