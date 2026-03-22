import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import warnings

# Core Logic Imports
from data_loader import clean_and_enhance_data
from engine_optimizer import train_role_models, calculate_role_agency
from metric_npv import calculate_npv
from metric_libr import calculate_libr

# --- CONFIGURATION ---
PROJECT_ROOT = Path("~/Desktop/Esports Project").expanduser()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AUDIT_FILE = OUTPUT_DIR / "Summer_Series_Prediction_Audit.csv"

def print_libr_leaderboard(df, top_n=10):
    """
    Groups by player to find seasonal performance and prints a clean leaderboard.
    """
    # Filter for player rows only and calculate seasonal average
    leaderboard = df.groupby(['playername', 'position'])['LIBR'].agg(['mean', 'count']).reset_index()
    leaderboard.columns = ['Player', 'Role', 'Avg_LIBR', 'Games_Played']
    
    # Filter for players with a meaningful sample size (e.g., > 5 games)
    leaderboard = leaderboard[leaderboard['Games_Played'] > 5]
    
    # Sort by the highest objective skill grade
    top_players = leaderboard.nlargest(top_n, 'Avg_LIBR')

    print("\n" + "-"*40)
    print(f"   TOP {top_n} PLAYERS BY OBJECTIVE LIBR   ")
    print("-"*40)
    print(f"{'Rank':<5} {'Player':<12} {'Role':<8} {'LIBR':<8}")
    
    for i, (_, row) in enumerate(top_players.iterrows(), 1):
        print(f"{i:<5} {row['Player']:<12} {row['Role']:<8} {row['Avg_LIBR']:>6.2f}")
    print("-"*40 + "\n")


class SeriesBacktest:
    """
    Simulates a betting/prediction environment.
    Trains on Spring data to predict Summer Series outcomes.
    """
    def __init__(self):
        self.predictor = LogisticRegression(fit_intercept=True)
        self.player_ratings = {}  # Stores player LIBR from Spring
        self.agency_map = {}      # Stores role impact weights
        
    def calculate_skill_gap(self, game_id, df_lineup, team_a_name):
        """
        Calculates the weighted skill difference between two teams for a specific game.
        """
        game_players = df_lineup[df_lineup['gameid'] == game_id]
        score_a = 0
        score_b = 0
        
        for _, p in game_players.iterrows():
            # Get player's skill from Spring; default to neutral (0) if new player
            rating = self.player_ratings.get(p['playername'], 0) 
            weight = self.agency_map.get(p['position'], 0)
            contribution = rating * weight
            
            if p['teamname'] == team_a_name:
                score_a += contribution
            else:
                score_b += contribution
                
        return score_a - score_b, score_a, score_b
    
    def run(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        print("\n" + "="*50)
        print("      LCK SUMMER SERIES BACKTEST STARTING      ")
        print("="*50 + "\n")
        
        # 1. Load Data
        df_players, _ = clean_and_enhance_data()
        df_spring = df_players[df_players['split'] == 'Spring'].copy()
        df_summer = df_players[df_players['split'] == 'Summer'].copy()
        
        # 2. SPRING PHASE: Knowledge Acquisition
        print("[1/4] Analyzing Spring Split to establish baselines...")
        models, scalers, features = train_role_models(df_spring)
        df_spring_scored = calculate_npv(df_spring, models, scalers, features)
        df_spring_scored = calculate_libr(df_spring_scored, models, features)
        
        # Map out the meta: How much does each role matter?
        raw_agency = calculate_role_agency(df_spring_scored)
        self.agency_map = {role: val/100 for role, val in raw_agency.items()}
        
        # Map out the players: How good is each individual?
        self.player_ratings = df_spring_scored.groupby('playername')['LIBR'].mean().to_dict()
        print_libr_leaderboard(df_spring_scored)

        # 3. CALIBRATION: Link Skill Gaps to Win Probabilities
        print("[2/4] Calibrating Skill Gap vs. Win Chance...")
        calibration_data = []
        for g_id in df_spring['gameid'].unique():
            game_subset = df_spring[df_spring['gameid'] == g_id]
            teams = game_subset.groupby('teamname')['result'].first().reset_index()
            
            if len(teams) == 2:
                team_a = teams.iloc[0]['teamname']
                diff, _, _ = self.calculate_skill_gap(g_id, df_spring, team_a)
                calibration_data.append({
                    'skill_diff': diff,
                    'team_a_win': int(teams.iloc[0]['result'])
                })
        
        df_calib = pd.DataFrame(calibration_data)
        self.predictor.fit(df_calib[['skill_diff']], df_calib['team_a_win'])

        # 4. SUMMER PHASE: The Prediction Test
        print("[3/4] Predicting Summer Series outcomes...")
        prediction_rows = []
        summer_game_ids = df_summer['gameid'].unique()
        
        for g_id in summer_game_ids:
            game_subset = df_summer[df_summer['gameid'] == g_id]
            teams = game_subset.groupby(['teamname', 'date', 'result']).first().reset_index()
            
            if len(teams) != 2: continue
            
            team_a = teams.iloc[0]
            team_b = teams.iloc[1]
            
            diff, _, _ = self.calculate_skill_gap(g_id, df_summer, team_a['teamname'])
            # Get probability that Team A wins
            prob_a = self.predictor.predict_proba(pd.DataFrame({'skill_diff': [diff]}))[0][1]
            
            prediction_rows.append({
                'date': team_a['date'],
                'gameid': g_id,
                'team_a': team_a['teamname'],
                'team_b': team_b['teamname'],
                'prob_a': prob_a,
                'result_a': team_a['result']
            })

        # 5. AGGREGATION: Turn individual games into Series (Bo3/Bo5)
        df_pred = pd.DataFrame(prediction_rows)
        # Create a unique ID for the series (Date + Sorted Team Names)
        df_pred['series_id'] = df_pred.apply(
            lambda x: f"{x['date'].split()[0]}_{'_'.join(sorted([x['team_a'], x['team_b']]))}", axis=1
        )
        
        final_results = []
        for _, group in df_pred.groupby('series_id'):
            avg_prob_a = group['prob_a'].mean()
            team_a_name = group.iloc[0]['team_a']
            team_b_name = group.iloc[0]['team_b']
            
            predicted_winner = team_a_name if avg_prob_a > 0.5 else team_b_name
            
            wins_a = group['result_a'].sum()
            wins_b = len(group) - wins_a
            actual_winner = team_a_name if wins_a > wins_b else team_b_name
            
            final_results.append({
                'Date': group.iloc[0]['date'].split()[0],
                'Matchup': f"{team_a_name} vs {team_b_name}",
                'Predicted_Winner': predicted_winner,
                'Confidence': f"{max(avg_prob_a, 1-avg_prob_a):.1%}",
                'Actual_Score': f"{wins_a}-{wins_b}" if team_a_name == actual_winner else f"{wins_b}-{wins_a}",
                'Correct': predicted_winner == actual_winner
            })

        # 6. OUTPUT
        if not OUTPUT_DIR.exists(): OUTPUT_DIR.mkdir(parents=True)
        df_audit = pd.DataFrame(final_results)
        df_audit.to_csv(AUDIT_FILE, index=False)
        
        accuracy = df_audit['Correct'].mean()
        print(f"\n[RESULTS] Summer Series Accuracy: {accuracy:.1%}")
        print(f"Audit Log saved to: {AUDIT_FILE}\n")

if __name__ == "__main__":
    backtester = SeriesBacktest()
    backtester.run()
