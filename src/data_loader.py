import pandas as pd
import numpy as np
from pathlib import Path

# Setup Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FILE_PATH = DATA_DIR / "2023_LoL_esports_match_data_from_OraclesElixir.csv"

# Configuration: Define stats for calculations outside the functions
NET_TARGET_STATS = [
    'golddiffat15', 'xpdiffat15', 'csdiffat15', 
    'golddiffat25', 'xpdiffat25', 'csdiffat25',
    'damagetochampions', 'earnedgold'
]

TEAM_MERGE_COLS = [
    'gameid', 'teamid', 'kills', 'deaths', 
    'damagetochampions', 'totalgold', 'earnedgold'
]

def calculate_net_features(df_players, target_stats):
    """
    Calculates the 'Net' performance (Player Stat - Team Average).
    """
    # Calculate Team Totals per game and map back to player rows
    team_sums = df_players.groupby(['gameid', 'teamid'])[target_stats].transform('sum')
    
    for stat in target_stats:
        df_players[f'net_{stat}'] = df_players[stat] - (team_sums[stat] / 5)
        
    return df_players

def clean_and_enhance_data():
    """Main data loading and preprocessing pipeline."""
    if not FILE_PATH.exists():
        raise FileNotFoundError(f"Data file not found at: {FILE_PATH}")

    df = pd.read_csv(FILE_PATH, low_memory=False)
    
    # 1. Filter for LCK and Complete Data
    df = df[(df['datacompleteness'] == 'complete') & (df['league'] == 'LCK')].copy()
    
    df_players = df[df['position'] != 'team'].copy()
    df_teams = df[df['position'] == 'team'].copy()

    # 2. Extract and Merge Team Totals
    team_stats = df_teams[TEAM_MERGE_COLS].copy()
    
    # Rename columns with prefix for clarity, keeping gameid and teamid as keys
    new_col_names = {col: f'team_{col}' for col in TEAM_MERGE_COLS if col not in ['gameid', 'teamid']}
    team_stats = team_stats.rename(columns=new_col_names)
    
    df_players = df_players.merge(team_stats, on=['gameid', 'teamid'], how='left')

    # 3. Calculate KP (Kill Participation)
    # Using replace(0, np.nan) to avoid ZeroDivisionError
    df_players['kp'] = (
        (df_players['kills'] + df_players['assists']) / 
        df_players['team_kills'].replace(0, np.nan)
    ).fillna(0)
    
    # 4. Inject Net Features using factored list
    df_players = calculate_net_features(df_players, NET_TARGET_STATS)
    
    return df_players, df_teams
