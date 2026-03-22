import numpy as np
import pandas as pd

# --- LEGACY WEIGHTS (Commented for Reproducibility) ---
# LIBR_WEIGHTS = {
#     'top': {'g_voa': 0.15, 'g_dom': 0.25, 'ncv': 0.15, 'vspm': 0.05, 'fb': 0.05, 'c_voa': 0.15, 'x_voa': 0.20},
#     'jng': {'g_voa': 0.15, 'g_dom': 0.25, 'ncv': 0.25, 'vspm': 0.10, 'fb': 0.10, 'c_voa': 0.05, 'x_voa': 0.10},
#     'mid': {'g_voa': 0.25, 'g_dom': 0.20, 'ncv': 0.175, 'vspm': 0.075,'fb': 0.05, 'c_voa': 0.15, 'x_voa': 0.10},
#     'bot': {'g_voa': 0.30, 'g_dom': 0.15, 'ncv': 0.10, 'vspm': 0.025,'fb': 0.025,'c_voa': 0.25, 'x_voa': 0.15},
#     'sup': {'g_voa': 0.05, 'g_dom': 0.15, 'ncv': 0.30, 'vspm': 0.30, 'fb': 0.09, 'c_voa': 0.01, 'x_voa': 0.10}
# }
# COMBAT_WEIGHTS = {
#     'top': {'k': 1.0, 'a': 0.5}, 'jng': {'k': 0.75, 'a': 0.75}, 'mid': {'k': 1.0, 'a': 0.5},
#     'bot': {'k': 1.0, 'a': 0.5}, 'sup': {'k': 0.5, 'a': 1.0}
# }


class ObjectiveLIBREngine:
    def __init__(self, df_players, models, features):
        """
        Calculates LIBR using weights discovered by the role-specific models.
        """
        self.df = df_players
        self.features = features
        self.dynamic_weights = self._extract_model_weights(models)
        self.baselines = self._calculate_baselines()

    def _extract_model_weights(self, models):
        """Discovers feature importance from Logistic Regression coefficients."""
        role_weights = {}
        for role, model in models.items():
            # Get absolute coefficients to represent 'importance'
            abs_coefs = np.abs(model.coef_[0])
            # Normalize so weights sum to 1.0 per role
            normalized = abs_coefs / np.sum(abs_coefs)
            role_weights[role] = dict(zip(self.features, normalized))
        return role_weights

    def _calculate_baselines(self):
        """Calculates role-specific Z-score anchors for all survivor features."""
        stats = {}
        roles = ['top', 'jng', 'mid', 'bot', 'sup']
        for role in roles:
            rdf = self.df[self.df['position'] == role].copy()
            role_stats = {}
            for feat in self.features:
                role_stats[f"{feat}_avg"] = rdf[feat].mean()
                role_stats[f"{feat}_std"] = rdf[feat].std() if rdf[feat].std() != 0 else 1.0
            stats[role] = role_stats
        return stats

    def calculate_score(self, row):
        role = row['position']
        if role not in self.dynamic_weights:
            return 0
        
        weights = self.dynamic_weights[role]
        baselines = self.baselines[role]
        
        total_score = 0
        for feat in self.features:
            # Step 1: Calculate the Z-score (Distance from Pro Average)
            z_val = (row[feat] - baselines[f"{feat}_avg"]) / baselines[f"{feat}_std"]
            
            # Step 2: Apply the weight discovered by the model
            # We scale by 10 to keep the score in a readable range (e.g., 0-15)
            total_score += (z_val * weights[feat]) * 10
                
        return round(total_score, 2)

def calculate_libr(df_players, models, features):
    """
    Unified entry point. 
    Note: Now requires the models and features from train_role_models().
    """
    engine = ObjectiveLIBREngine(df_players, models, features)
    df_players['LIBR'] = df_players.apply(engine.calculate_score, axis=1)
    return df_players

