import numpy as np
import pandas as pd

# Descriptive Weights (Subjective/Scouting based)
LIBR_WEIGHTS = {
    'top': {'g_voa': 0.15, 'g_dom': 0.25, 'ncv': 0.15, 'vspm': 0.05, 'fb': 0.05, 'c_voa': 0.15, 'x_voa': 0.20},
    'jng': {'g_voa': 0.15, 'g_dom': 0.25, 'ncv': 0.25, 'vspm': 0.10, 'fb': 0.10, 'c_voa': 0.05, 'x_voa': 0.10},
    'mid': {'g_voa': 0.25, 'g_dom': 0.20, 'ncv': 0.175, 'vspm': 0.075,'fb': 0.05, 'c_voa': 0.15, 'x_voa': 0.10},
    'bot': {'g_voa': 0.30, 'g_dom': 0.15, 'ncv': 0.10, 'vspm': 0.025,'fb': 0.025,'c_voa': 0.25, 'x_voa': 0.15},
    'sup': {'g_voa': 0.05, 'g_dom': 0.15, 'ncv': 0.30, 'vspm': 0.30, 'fb': 0.09, 'c_voa': 0.01, 'x_voa': 0.10}
}

COMBAT_WEIGHTS = {
    'top': {'k': 1.0, 'a': 0.5}, 'jng': {'k': 0.75, 'a': 0.75}, 'mid': {'k': 1.0, 'a': 0.5},
    'bot': {'k': 1.0, 'a': 0.5}, 'sup': {'k': 0.5, 'a': 1.0}
}

class LIBREngine:
    def __init__(self, df_players):
        self.df = df_players
        self.baselines = self._calculate_baselines()

    def _calculate_baselines(self):
        """Calculates Role Averages for VOA (Value Over Average) and Z-Scores."""
        stats = {}
        for role in LIBR_WEIGHTS.keys():
            rdf = self.df[self.df['position'] == role].copy()
            w = COMBAT_WEIGHTS[role]
            rdf['ncv'] = (rdf['killsat15'] * w['k'] + rdf['assistsat15'] * w['a']) - rdf['deathsat15']
            
            stats[role] = {
                'gold_avg': rdf['earnedgold'].mean(),
                'xp_avg': rdf['xpdiffat15'].mean(), # using diff as proxy for laning strength
                'cs_avg': rdf['cspm'].mean(),
                'ncv_avg': rdf['ncv'].mean(), 'ncv_std': rdf['ncv'].std(),
                'vspm_avg': rdf['vspm'].mean(), 'vspm_std': rdf['vspm'].std(),
                'gd15_std': rdf['golddiffat15'].std()
            }
        return stats

    def calculate_score(self, row):
        role = row['position']
        if role not in LIBR_WEIGHTS: return 0
        
        b = self.baselines[role]
        w = LIBR_WEIGHTS[role]
        cw = COMBAT_WEIGHTS[role]
        
        # 1. VOA (Value Over Average in %)
        g_voa = (row['earnedgold'] / b['gold_avg'] * 100) - 100
        c_voa = (row['cspm'] / b['cs_avg'] * 100) - 100
        # XP VOA is tricky, we'll simplify to scaled diff for stability
        x_voa = row['xpdiffat15'] * 0.1 

        # 2. Dominance (Sigma-Normalized)
        g_dom = (row['golddiffat15'] / b['gd15_std']) * 10
        
        # 3. Combat
        ncv_raw = (row['killsat15'] * cw['k'] + row['assistsat15'] * cw['a']) - row['deathsat15']
        ncv_score = ((ncv_raw - b['ncv_avg']) / b['ncv_std']) * 10
        
        # 4. Utility
        vspm_score = ((row['vspm'] - b['vspm_avg']) / b['vspm_std']) * 10
        fb_score = (row['firstbloodkill'] + row['firstbloodassist']*0.5 - row['firstbloodvictim']) * 10

        total = (g_voa * w['g_voa']) + (c_voa * w['c_voa']) + (x_voa * w['x_voa']) + \
                (g_dom * w['g_dom']) + (ncv_score * w['ncv']) + (vspm_score * w['vspm']) + (fb_score * w['fb'])
                
        return round(total, 2)

def calculate_libr(df_players):
    engine = LIBREngine(df_players)
    df_players['LIBR'] = df_players.apply(engine.calculate_score, axis=1)
    return df_players