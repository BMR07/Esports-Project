import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import warnings

# Import Core Logic
from data_loader import clean_and_enhance_data
from engine_optimizer import train_role_models, calculate_role_agency
from metric_npv import calculate_npv
from metric_libr import calculate_libr

# --- CONFIGURATION ---
PROJECT_ROOT = Path("~/Desktop/Esports Project").expanduser()
AUDIT_FILE = PROJECT_ROOT / "Summer_Series_Prediction_Audit.csv"

class SeriesBacktest:
    def __init__(self):
        self.predictor = LogisticRegression(fit_intercept=True)
        self.player_ratings = {} 
        self.agency_map = {}
        self.audit_log = []

    def get_matchup_diff(self, game_row, df_lineup):
        """Calculates Skill Gap based on Spring Ratings."""
        game_players = df_lineup[df_lineup['gameid'] == game_row['gameid']]
        score_blue = 0
        score_red = 0
        
        for _, p in game_players.iterrows():
            rating = self.player_ratings.get(p['playername'], 0) 
            weight = self.agency_map.get(p['position'], 0)
            val = rating * weight
            
            if p['teamname'] == game_row['team_a']:
                score_blue += val
            else:
                score_red += val
                
        return score_blue - score_red, score_blue, score_red

    def run(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        print("--- STARTING SERIES-LEVEL BACKTEST ---")
        
        # 1. Load Data
        print("[1/6] Loading Data...")
        df_players, _ = clean_and_enhance_data()
        
        df_spring = df_players[df_players['split'] == 'Spring'].copy()
        df_summer = df_players[df_players['split'] == 'Summer'].copy()
        
        # 2. SPRING PHASE (Training)
        print("[2/6] Generating Historical Ratings (Spring)...")
        models, scalers, features = train_role_models(df_spring)
        df_spring_scored = calculate_npv(df_spring, models, scalers, features)
        df_spring_scored = calculate_libr(df_spring_scored)
        
        raw_agency = calculate_role_agency(df_spring_scored)
        self.agency_map = {k: v/100 for k, v in raw_agency.items()}
        self.player_ratings = df_spring_scored.groupby('playername')['LIBR'].mean().to_dict()
        
        print("      > Spring Agency (Meta):", {k: f"{v:.1%}" for k, v in self.agency_map.items()})

        # 3. Calibration (Game Level)
        print("[3/6] Calibrating Probability Model...")
        spring_games = []
        spring_match_ids = df_spring['gameid'].unique()
        
        for g_id in spring_match_ids:
            teams = df_spring[df_spring['gameid'] == g_id].groupby('teamname')['result'].first().reset_index()
            if len(teams) != 2: continue
            
            row_dummy = {'gameid': g_id, 'team_a': teams.iloc[0]['teamname']}
            diff, _, _ = self.get_matchup_diff(row_dummy, df_spring)
            
            spring_games.append({
                'skill_diff': diff,
                'team_a_win': int(teams.iloc[0]['result'])
            })
            
        df_calib = pd.DataFrame(spring_games)
        self.predictor.fit(df_calib[['skill_diff']], df_calib['team_a_win'])

        # 4. SUMMER PHASE (Series Prediction)
        print("[4/6] Predicting Summer Series...")
        
        # We group by Date + Teams to identify a "Series"
        summer_games = df_summer.groupby(['gameid', 'teamname', 'result', 'date']).first().reset_index()
        
        game_list = []
        unique_games = summer_games['gameid'].unique()
        for g_id in unique_games:
            teams = summer_games[summer_games['gameid'] == g_id]
            if len(teams) != 2: continue
            team_a = teams.iloc[0]
            team_b = teams.iloc[1]
            
            diff, score_a, score_b = self.get_matchup_diff({'gameid': g_id, 'team_a': team_a['teamname']}, df_summer)
            prob_a_win = self.predictor.predict_proba(pd.DataFrame({'skill_diff': [diff]}))[0][1]
            
            game_list.append({
                'date': team_a['date'], 
                'gameid': g_id,
                'team_a': team_a['teamname'], 
                'team_b': team_b['teamname'], 
                'prob_a': prob_a_win,
                'result_a': team_a['result']
            })
            
        df_summer_games = pd.DataFrame(game_list)
        
        # --- FIX: SPLIT DATE TO REMOVE TIME ---
        # This groups "2023-06-07 15:00" and "2023-06-07 16:00" into the same series ID
        df_summer_games['series_id'] = df_summer_games.apply(
            lambda x: f"{x['date'].split()[0]}_{'_'.join(sorted([x['team_a'], x['team_b']]))}", axis=1
        )
        
        series_results = []
        for s_id, group in df_summer_games.groupby('series_id'):
            # Static prediction for the series (average of games)
            avg_prob_a = group['prob_a'].mean()
            predicted_winner = group.iloc[0]['team_a'] if avg_prob_a > 0.5 else group.iloc[0]['team_b']
            
            # Actual Winner: Count wins in this group
            wins_a = group['result_a'].sum()
            total_games = len(group)
            wins_b = total_games - wins_a
            
            actual_winner = group.iloc[0]['team_a'] if wins_a > wins_b else group.iloc[0]['team_b']
            
            # Formating Score safely for CSV (Avoiding Jan-00 Excel issues)
            score_str = f"'{wins_a}-{wins_b}" if predicted_winner == group.iloc[0]['team_a'] else f"'{wins_b}-{wins_a}"
            
            series_results.append({
                'Date': group.iloc[0]['date'].split()[0],
                'Series': f"{group.iloc[0]['team_a']} vs {group.iloc[0]['team_b']}",
                'Predicted_Winner': predicted_winner,
                'Win_Probability': round(avg_prob_a if avg_prob_a > 0.5 else 1-avg_prob_a, 3),
                'Actual_Winner': actual_winner,
                'Correct': predicted_winner == actual_winner,
                'Score': score_str # Added tick to force string in Excel
            })

        # 5. EXPORT & RESULTS
        df_audit = pd.DataFrame(series_results)
        df_audit.to_csv(AUDIT_FILE, index=False)
        
        acc = df_audit['Correct'].mean()
        print(f"\n--- FINAL SERIES ACCURACY ---")
        print(f"Series Correct: {df_audit['Correct'].sum()} / {len(df_audit)}")
        print(f"Accuracy: {acc:.1%}")
        print(f"Saved to: {AUDIT_FILE}")

if __name__ == "__main__":
    test = SeriesBacktest()
    test.run()