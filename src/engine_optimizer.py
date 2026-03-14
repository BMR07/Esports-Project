import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# The "Survivor" Feature Set: Key performance indicators with low multicollinearity.
SURVIVOR_FEATURES = [
    'net_golddiffat15', 'net_xpdiffat15', 'net_csdiffat15',
    'net_golddiffat25', 'net_xpdiffat25', 'net_csdiffat25',
    'dpm', 'damageshare', 'earnedgoldshare', 'kp',
    'cspm', 'vspm', 'wpm', 'kills', 'deaths', 'assists'
]

def train_role_models(df_players):
    """
    STAGES 1 & 2: Analyzes role factors and evaluates players.
    Trains five separate Logistic Regression models to define what 'good' 
    performance looks like for each specific role.
    """
    models = {}
    scalers = {}
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    
    for role in roles:
        # Filter for role and ensure no missing values in features or target
        role_df = df_players[df_players['position'] == role].dropna(subset=SURVIVOR_FEATURES + ['result'])
        
        X = role_df[SURVIVOR_FEATURES]
        y = role_df['result']
        
        # Scale features: Essential for Logistic Regression coefficients to be comparable
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit model: This determines which SURVIVOR_FEATURES matter most for this role
        model = LogisticRegression(max_iter=2000, C=1.0) 
        model.fit(X_scaled, y)
        
        models[role] = model
        scalers[role] = scaler
        
    return models, scalers, SURVIVOR_FEATURES

def calculate_role_agency(df_w_probs):
    """
    STAGE 3: Evaluates role importance (Agency).
    Uses the raw win probabilities (player grades) of all 5 teammates to 
    predict the game outcome. The resulting weights represent 'Role Agency'.
    """
    # 1. Pivot data to represent one team's performance per row
    # Required format: [gameid, teamname, result, top, jng, mid, bot, sup]
    agency_df = df_w_probs.pivot_table(
        index=['gameid', 'teamname', 'result'], 
        columns='position', 
        values='p_win_raw'
    ).reset_index().dropna()
    
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    X = agency_df[roles]
    y = agency_df['result']
    
    # 2. Fit the Meta-Model
    # We use fit_intercept=False because if all players have 0 probability, 
    # the team should have 0 probability of winning.
    meta_model = LogisticRegression(fit_intercept=False)
    meta_model.fit(X, y)
    
    # 3. Extract and Normalize Weights
    # Coefficients tell us how much a 1% increase in a player's performance 
    # impacts the team's total win chance.
    raw_weights = np.abs(meta_model.coef_[0])
    normalized_weights = (raw_weights / raw_weights.sum()) * 100
    
    return dict(zip(roles, normalized_weights))
