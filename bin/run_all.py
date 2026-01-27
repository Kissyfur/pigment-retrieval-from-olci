import subprocess
import sys

commands = [
    # Experiments OLCI Talone dataset (very long)
    # TODO: run again when there is time to check changes do not affect
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/OLCI.json"],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI.json"],

    # Experiments OLCI matchup dataset
    [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/OLCI_sat.json"],
    [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI_sat.json"],

    # Experiments OLCI matchup dataset fine tuning
    [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/OLCI_sat_preft.json"],
    [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI_sat_preft.json"],
    [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI_sat_ft.json"],
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
