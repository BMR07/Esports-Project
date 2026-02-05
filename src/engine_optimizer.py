import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# The "Survivor" Feature Set (r > 0.70 pruned)
SURVIVOR_FEATURES = [
    'net_golddiffat15', 'net_xpdiffat15', 'net_csdiffat15',
    'net_golddiffat25', 'net_xpdiffat25', 'net_csdiffat25',
    'dpm', 'damageshare', 'earnedgoldshare', 'kp',
    'cspm', 'vspm', 'wpm', 'kills', 'deaths', 'assists'
]

def train_role_models(df_players):
    """Trains a Logistic Regression for each role to predict Wins."""
    models = {}
    scalers = {}
    
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    
    for role in roles:
        role_df = df_players[df_players['position'] == role].dropna(subset=SURVIVOR_FEATURES + ['result'])
        X = role_df[SURVIVOR_FEATURES]
        y = role_df['result']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # High max_iter to ensure convergence
        model = LogisticRegression(max_iter=2000)
        model.fit(X_scaled, y)
        
        models[role] = model
        scalers[role] = scaler
        
    return models, scalers, SURVIVOR_FEATURES

def calculate_role_agency(df_w_probs):
    """
    Runs the Meta-Model: Uses player Win Probabilities to predict the final Game Result.
    Coefficients determine how much 'Agency' each role has.
    """
    # Pivot to get one row per game: [gameid, top_p, jng_p, ..., result]
    agency_df = df_w_probs.pivot_table(
        index=['gameid', 'teamname', 'result'], 
        columns='position', 
        values='p_win_raw'
    ).reset_index().dropna()
    
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    X = agency_df[roles]
    y = agency_df['result']
    
    meta_model = LogisticRegression(fit_intercept=False) # No intercept, purely weight based
    meta_model.fit(X, y)
    
    # Normalize coefficients to percentages
    weights = np.abs(meta_model.coef_[0])
    agency_map = dict(zip(roles, (weights / weights.sum()) * 100))
    
    return agency_map