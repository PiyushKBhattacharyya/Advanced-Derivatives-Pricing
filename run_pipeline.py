import os
import subprocess
import sys

# Workaround for OpenMP duplication error (libomp.dylib conflict)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"Executing: src/{script_name}")
    print(f"{'='*60}")
    
    # Use the local virtual environment Python executable natively
    python_exe = sys.executable
    
    # Run as a native module to prevent 'src' import path errors
    module_name = f"src.{script_name[:-3]}"
    result = subprocess.run([python_exe, "-m", module_name])
    
    if result.returncode != 0:
        print(f"\n[ERROR] Pipeline execution failed natively at: {script_name}")
        print("Please check the error trace above and resolve the Python exception.")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] {script_name} completed cleanly.")

if __name__ == "__main__":
    print("============================================================")
    print("   DEEP BSDE QUANTITATIVE PRICING PIPELINE INITIALIZATION   ")
    print("============================================================")
    
    # The strictly enforced sequence required to rebuild all mathematical states from scratch
    execution_sequence = [
        "data_loader.py",            # 1. Pull the latest Live Options Chains from the external broker/API
        "market_paths.py",           # 2. Extract strictly 15-30 years of True S&P 500 & VIX trajectories 
        "institutional_baselines.py",# 3. Generate SABR / rBergomi benchmarks for Tier-1 comparison
        "stress_test_gen.py",        # 4. Phase 7: Generate Synthetic Shadow Crashes via MarketVAE
        "train.py",                  # 5. Train the Friction-Aware Neural Architecture
        "rl_agent.py",               # 6. Train the Frictional RL Pipeline against Empirical Data
        "explainability.py",         # 7. Phase 7: Execute Feature Attribution / Saliency maps
        "validation.py",             # 8. Numerically validate latency, COVID-19 and Synthetic Crashes
        "generate_figures.py"        # 9. Final Quant Armoury visualization compilation
    ]
    
    for script in execution_sequence:
        run_script(script)
        
    print("\n" + "="*60)
    print("PIPELINE FULLY COMPLETE!")
    print("Review the outputs in the Data/ and Figs/ directories.")
    print("="*60)
