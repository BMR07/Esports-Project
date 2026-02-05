# init_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from data_clean import FILE_PATH, clean_and_enhance_data, create_games_df, filter_to_lck

# Global Constants for 2023
PASSIVE_GPM = 122.4
STARTING_GOLD = 500
TIME_POINTS = np.array([10, 15, 20, 25])

def get_earned_gold(total_gold_array, times=TIME_POINTS):
    """Helper to convert raw gold milestones to earned gold."""
    return total_gold_array - STARTING_GOLD - (PASSIVE_GPM * times)

def fit_snowball_curve(times, earned_gold):
    """Fits a degree-2 polynomial to gold data."""
    poly = PolynomialFeatures(degree=2)
    times_reshaped = times.reshape(-1, 1)
    X_poly = poly.fit_transform(times_reshaped)
    model = LinearRegression().fit(X_poly, earned_gold)
    return model, poly

def calculate_positional_benchmarks(df_players):
    benchmarks = df_players.groupby('position').agg({
        'cspm': ['mean', 'std'],
        'earned gpm': ['mean', 'std']
    }).reset_index()
    benchmarks.columns = ['position', 'cspm_mean', 'cspm_std', 'gpm_mean', 'gpm_std']
    return benchmarks

def plot_positional_gold_progression(df_players):
    pos_order = ['top', 'jng', 'mid', 'bot', 'sup']
    positions = [p for p in pos_order if p in df_players['position'].unique()]
    models = {}
    fig, axes = plt.subplots(1, len(positions), figsize=(25, 6), sharey=True)

    for i, pos in enumerate(positions):
        pos_df = df_players[df_players['position'] == pos]
        gold_means = np.array([
            pos_df['goldat10'].mean(), pos_df['goldat15'].mean(),
            pos_df['goldat20'].mean(), pos_df['goldat25'].dropna().mean() 
        ])
        
        earned_gold = get_earned_gold(gold_means)
        model, poly = fit_snowball_curve(TIME_POINTS, earned_gold)
        models[pos] = (model, poly)
        
        # Plotting logic
        x_range = np.linspace(10, 25, 100).reshape(-1, 1)
        axes[i].scatter(TIME_POINTS, earned_gold, color='black')
        axes[i].plot(x_range, model.predict(poly.transform(x_range)), 'b-')
        axes[i].set_title(f'Role: {pos.upper()}')
        
    plt.savefig('lck_positional_snowball_curves.png')
    return models

if __name__ == "__main__":
    df_players_all, df_teams_all = clean_and_enhance_data(FILE_PATH)
    df_lck_players, _, _ = filter_to_lck(df_players_all, df_teams_all, create_games_df(df_teams_all))
    role_models = plot_positional_gold_progression(df_lck_players)
    print("--- LCK 2023 Global Analysis Complete ---")