import pandas as pd
import numpy as np
from pathlib import Path

# Module Imports
from data_loader import clean_and_enhance_data
from engine_optimizer import train_role_models, calculate_role_agency
from metric_npv import calculate_npv
from metric_libr import calculate_libr

# --- CONFIGURATION ---
# Using .expanduser() ensures it works regardless of the specific OS user
PROJECT_ROOT = Path("~/Desktop/Esports Project").expanduser()
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "LIBR_Final_Output.csv"

def run_project_pipeline():
    """
    Orchestrates the LIBR (League Impact Benchmark Rating) pipeline.
    
    This follows a two-pass architecture:
    1. Discovery: Training models and calculating raw win probabilities.
    2. Application: Weighting those probabilities by role agency to find 
       true individual impact.
    """
    print("\n" + "="*50)
    print("      PROJECT LIBR: DYNAMIC ANALYTICS PIPELINE      ")
    print("="*50 + "\n")
    
    # 0. Infrastructure Check
    if not OUTPUT_DIR.exists():
        print(f"[0/6] Creating output directory: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("[1/6] Loading & Cleaning LCK Data...")
    df_players, df_teams = clean_and_enhance_data()

    # 2. Train Engine
    print("[2/6] Training Logistic Regression Role Engines...")
    models, scalers, features = train_role_models(df_players)
    
    # 3. Pass 1: Raw NPV (Discovery)
    # Calculates p_win_raw: The unweighted probability of winning based on stats.
    print("[3/6] Pass 1: Calculating Raw Win Probabilities (NPV)...")
    df_pass_1 = calculate_npv(df_players, models, scalers, features, agency_map=None)

    # 4. Calculate Role Agency (The Learning Step)
    # Determines which factors in each role actually correlate most with winning.
    print("[4/6] Discovered Role Agency Weights:")
    agency_map = calculate_role_agency(df_pass_1)
    for role, weight in agency_map.items():
        print(f"      > {role:7}: {weight:.1%}")

    # 5. Pass 2: Weighted NPV (Application)
    # Applies the agency weights to the raw probabilities for a 'True Impact' score.
    print("[5/6] Pass 2: Applying Agency Weights to NPV...")
    df_final = calculate_npv(df_players, models, scalers, features, agency_map=agency_map)

    # 6. Calculate LIBR (Skill Index)
    # Final descriptive metric for ranking and leaderboard visualization.
    print("[6/6] Finalizing LIBR Skill Index...")
    df_final = calculate_libr(df_final)
    
    # Output Results
    print(f"\n[COMPLETE] Saving results to: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("Pipeline finished successfully.\n")

if __name__ == "__main__":
    run_project_pipeline()
