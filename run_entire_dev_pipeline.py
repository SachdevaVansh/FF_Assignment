import subprocess
from pathlib import Path
import sys
import os

import time

def main():
    # Step 1: Run the setup script
    setup_script = "create_virtual_env.py"

    print(f"Step 1: Running setup script: {setup_script}")
    result = subprocess.run([sys.executable, setup_script], capture_output=True, text=True)

    if result.returncode != 0:
        print("Setup script failed.")
        print(result.stderr)
        sys.exit(1)

    # Formula used to construct venv_name in setup_script aka 'create_virtual_env.py'
    venv_name = "venv_" + Path(os.getcwd()).stem.lower()

    # Step 2: Run training script in venv
    print(f"\n Step 2: Running training script in venv...")
    venv_python_exe = Path(venv_name) / "Scripts" / "python.exe"
    print(f"Preparing Image Data for Training\n")

    # Preprocess portion
    result_preprocess = subprocess.run([str(venv_python_exe), "setup_training.py"])
    if result_preprocess.returncode != 0:
        print("Preprocessing failed...")
    else:
        print("Processing successful!")

    # Train portion
    print(f"Training the Model...\n")
    result_training = subprocess.run([str(venv_python_exe), "train_model.py"])
    if result_training.returncode != 0:
        print("Training failed...")
    else:
        print("Training successful...")

if __name__ == "__main__":
    from datetime import datetime
    start_time = time.time()
    dt = datetime.fromtimestamp(start_time)
    print(f"Time begins now {dt}")

    main()

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"Execution time: {minutes} min {seconds:.2f} sec")