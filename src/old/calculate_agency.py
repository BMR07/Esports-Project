import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from calculate_npv import get_trained_models, calculate_zero_sum_npv
from data_clean import FILE_PATH, clean_and_enhance_data, filter_to_lck, create_games_df, calculate_net_features

def solve_role_agency(df_final):
    # 1. Pivot the data so each row is ONE team performance in ONE game
    # Columns: gameid, teamname, result, top_p, jng_p, mid_p, bot_p, sup_p
    agency_df = df_final.pivot_table(
        index=['gameid', 'teamname', 'result'], 
        columns='position', 
        values='p_win_raw'
    ).reset_index()
    
    # Drop any games where we don't have all 5 positions (e.g., data errors)
    agency_df = agency_df.dropna(subset=['top', 'jng', 'mid', 'bot', 'sup'])
    
    # 2. Define Features (Role Probabilities) and Target (Actual Result)
    roles = ['top', 'jng', 'mid', 'bot', 'sup']
    X = agency_df[roles]
    y = agency_df['result']
    
    # 3. Fit the Meta-Model
    # We don't scale here because all inputs are already on the 0-1 probability scale
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X, y)
    
    # 4. Extract Coefficients as Agency Weights
    weights = meta_model.coef_[0]
    
    # Normalize weights so they sum to 100% for easy interpretability
    normalized_weights = (np.abs(weights) / np.abs(weights).sum()) * 100
    
    return dict(zip(roles, np.round(normalized_weights, 2)))

if __name__ == "__main__":
    # Load and Prepare
    df_p, df_t = clean_and_enhance_data(FILE_PATH)
    df_lck, df_t_all, _ = filter_to_lck(df_p, df_t, create_games_df(df_t))
    df_net = calculate_net_features(df_lck, df_t_all)
    
    # Get the raw probabilities per player
    models, scalers, features = get_trained_models(df_net)
    df_final = calculate_zero_sum_npv(df_net, models, scalers, features)
    
    # Run the Agency Meta-Model
    print("Calculating Role Agency Weights...")
    agency_map = solve_role_agency(df_final)
    
    print("\n--- 2023 LCK Role Agency (Impact on Win %) ---")
    sorted_agency = sorted(agency_map.items(), key=lambda x: x[1], reverse=True)
    for role, weight in sorted_agency:
        print(f"{role.upper():<5}: {weight}%")