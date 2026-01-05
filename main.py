import subprocess
import sys
import time
import os

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

PIPELINE_A = [
    "src/venture_scope/ingest/loaders_enriched.py",
    "src/venture_scope/features/kpi.py",
    "src/venture_scope/features/scoring.py",
    "src/venture_scope/ml/model.py",
    "examples/create_visualizations_v2.py",
    "examples/missing_data_analysis.py"
]

PIPELINE_B = [
    "scripts/temporal_split.py",
    "src/venture_scope/ml/model_temporal.py",
    "scripts/compare_models.py",
    "scripts/error_analysis.py",
    "scripts/distribution_shift_analysis.py",
    "src/venture_scope/ml/model_comparison.py",
    "tests/test_temporal_split.py"
]

PREDICTION = [
    "src/venture_scope/ml/predict.py"
]

# ==============================================================================
# EXECUTION ENGINE
# ==============================================================================

def run_script(script_path, interactive=False):
    """
    Exécute un script Python en affichant la sortie en temps réel dans le terminal.
    """
    # Auto-découverte du fichier
    if not os.path.exists(script_path):
        found = False
        filename = os.path.basename(script_path)
        for root, dirs, files in os.walk("."):
            if filename in files:
                script_path = os.path.join(root, filename)
                found = True
                break
        
        if not found:
            print(f"ERROR: Le fichier '{script_path}' est introuvable.")
            return False

    print(f"\n--- Running: {os.path.basename(script_path)} ---")
    start_time = time.time()
    
    try:
        # Exécution sans capture de sortie : les logs s'affichent en direct
        result = subprocess.run(
            [sys.executable, script_path],
            check=False
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f">>> SUCCESS ({duration:.2f}s)")
            return True
        else:
            print(f">>> FAILURE ({duration:.2f}s) - Exit Code: {result.returncode}")
            return False

    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False

def run_pipeline(name, script_list, interactive_mode=False):
    """Exécute une liste séquentielle de scripts."""
    print(f"\n{'='*60}")
    print(f"STARTING PIPELINE {name}")
    print(f"{'='*60}")
    
    total_start = time.time()
    
    for script in script_list:
        
        if not run_script(script, interactive=interactive_mode):
            print(f"\nCRITICAL STOP in PIPELINE {name} at step '{script}'")
            return False
            
    total_duration = time.time() - total_start
    print(f"\nPIPELINE {name} COMPLETED ({total_duration:.2f}s)")
    return True

def wait_for_user(next_step_name):
    """
    Met le programme en pause jusqu'à ce que l'utilisateur appuie sur Entrée.
    Permet de lire les logs précédents avant de continuer.
    """
    print("\n" + "#"*60)
    print(f"   PAUSE : Press [ENTER] to start {next_step_name}...")
    print("#"*60)
    try:
        input()
    except KeyboardInterrupt:
        print("\n Execution stopped by user.")
        sys.exit(0)

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    # 1. Pipeline A (Batch)
    if run_pipeline("A (Data & Viz)", PIPELINE_A, interactive_mode=False):
        wait_for_user("Pipeline B (Modeling & Validation)")
        
        # 2. Pipeline B (Batch)
        if run_pipeline("B (Modeling & Validation)", PIPELINE_B, interactive_mode=False):
            wait_for_user("PREDICTION Phase")

            # 3. Prediction (INTERACTIF)
            
            if run_pipeline("PREDICTION", PREDICTION, interactive_mode=True):
                print("\nALL SYSTEMS GO: Pipelines A, B, and Prediction executed successfully.")
                sys.exit(0)
            else:
                print("\nExecution stopped at Prediction stage.")
                sys.exit(1)
        else:
            print("\nExecution stopped at Pipeline B.")
            sys.exit(1)

    else:
        print("\nGlobal execution interrupted due to an error in Pipeline A.")
        sys.exit(1)