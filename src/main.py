import pandas as pd
import numpy as np
from pathlib import Path
import os

# Import your modules
from data_loader import clean_and_enhance_data
from engine_optimizer import train_role_models, calculate_role_agency
from metric_npv import calculate_npv
from metric_libr import calculate_libr

# --- CONFIGURATION ---
PROJECT_ROOT = Path("~/Desktop/Esports Project").expanduser()
OUTPUT_FILE = PROJECT_ROOT / "LIBR_Final_Output.csv"

def run_project_libr():
    print("--- PROJECT LIBR: INITIALIZING DYNAMIC PIPELINE ---")
    
    # Ensure directory exists
    if not PROJECT_ROOT.exists():
        print(f"Creating directory: {PROJECT_ROOT}")
        PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("[1/6] Loading & Cleaning LCK Data...")
    df_players, df_teams = clean_and_enhance_data()

    # 2. Train Engine (Raw)
    print("[2/6] Training Role Engines (Logistic Regression)...")
    models, scalers, features = train_role_models(df_players)
    
    # 3. Pass 1: Raw NPV (Discovery)
    print("[3/6] Pass 1: Calculating Raw Probabilities...")
    # This generates 'p_win_raw', which is our key PREDICTIVE signal
    df_pass_1 = calculate_npv(df_players, models, scalers, features, agency_map=None)

    # 4. Calculate Role Agency (The Learning Step)
    print("[4/6] Learning Role Agency Weights from Data...")
    agency_map = calculate_role_agency(df_pass_1)
    print("      > Discovered Weights:", {k: f"{v:.1f}%" for k, v in agency_map.items()})

    # 5. Pass 2: Weighted NPV (Application)
    print("[5/6] Pass 2: Recalculating NPV with Discovered Weights...")
    df_final = calculate_npv(df_players, models, scalers, features, agency_map=agency_map)

    # 6. Calculate LIBR (Skill Index)
    # We still calculate this for the descriptive leaderboard, 
    # but Prediction Lab will ignore it in favor of NPV probabilities.
    print("[6/6] Calculating LIBR Skill Index...")
    df_final = calculate_libr(df_final)
    
    # OUTPUT
    print(f"Saving data to: {OUTPUT_FILE}")
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("\nPipeline Complete.")

if __name__ == "__main__":
    run_project_libr()