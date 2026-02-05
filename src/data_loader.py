import pandas as pd
import numpy as np
from pathlib import Path

# Setup Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FILE_PATH = DATA_DIR / "2023_LoL_esports_match_data_from_OraclesElixir.csv"

def calculate_net_features(df_players):
    """
    Implements the 'Teammate-Neutral' philosophy.
    Calculates the delta between player performance and the team average (1/5th team total).
    """
    target_stats = [
        'golddiffat15', 'xpdiffat15', 'csdiffat15', 
        'golddiffat25', 'xpdiffat25', 'csdiffat25',
        'damagetochampions', 'earnedgold'
    ]
    
    # Calculate Team Totals per game
    team_sums = df_players.groupby(['gameid', 'teamid'])[target_stats].transform('sum')
    
    # Apply Net Formula: Player - (Team_Sum / 5)
    for stat in target_stats:
        df_players[f'net_{stat}'] = df_players[stat] - (team_sums[stat] / 5)
        
    return df_players

def clean_and_enhance_data():
    """Main loading function."""
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Data file not found at: {FILE_PATH}")

    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # 1. Filter immediately for LCK & Complete Data
    df = df[(df['datacompleteness'] == 'complete') & (df['league'] == 'LCK')].copy()
    
    df_players = df[df['position'] != 'team'].copy()
    df_teams = df[df['position'] == 'team'].copy()

    # 2. Merge Team Totals for Rate Stats
    team_cols = ['gameid', 'teamname', 'kills', 'deaths', 'damagetochampions', 'totalgold', 'earnedgold']
    team_stats = df_teams[team_cols].copy()
    team_stats.columns = ['gameid', 'teamname', 'team_kills', 'team_deaths', 'team_damage', 'team_total_gold', 'team_earned_gold']
    
    df_players = df_players.merge(team_stats, on=['gameid', 'teamname'], how='left')

    # 3. Calculate Rate Stats (The "Survivors")
    df_players['kp'] = ((df_players['kills'] + df_players['assists']) / df_players['team_kills']).fillna(0)
    
    # 4. Inject Net Features
    df_players = calculate_net_features(df_players)
    
    return df_players, df_teams