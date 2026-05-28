import subprocess
import sys

METS = ["compute_metrics"]
ALL = ["train_modules", "train_models", "compute_metrics"]
MODELS_METS = ["train_models", "compute_metrics"]
MODULES = ["train_modules"]

#  = ["compute_metrics"]

commands = [
    # LONGEST EXPERIMENTS

    #                          Experiments OLCI Talone dataset (very long)
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/OLCI.json"],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI.json",
    #  "--steps", *STEPS],

    #                            Experiments multi Talone dataset (very long)
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/multi.json"],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/multi.json",
    #  "--steps", *STEPS],

    #                             Experiments Gonzalo dataset (very long)
    [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/gonzalo.json"],
    [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/gonzalo.json",
     "--steps", *ALL],

    #                             Experiments Gonzalo OLCI dataset (very long)
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/gonzalo_OLCI.json"],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/gonzalo_OLCI.json",
    #  "--steps", *ALL],

    #                                       SAT experiments
    # Experiments OLCI matchup dataset
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/OLCI_sat.json"],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI_sat.json",
    #  "--steps", *STEPS],

    # Experiments OLCI matchup dataset fine tuning
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/OLCI_sat_preft.json"],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI_sat_preft.json",
    #  "--steps", *MODULES],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI_sat_ft.json",
    #  "--steps", *MODELS_METS],

    # Experiments multi matchup dataset from scratch
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/OLCI_sat_scratch.json",
    #  "--steps", *MODELS_METS],



    #                                 Experiments multi matchup dataset
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/multi_sat.json"],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/multi_sat.json",
    #  "--steps", *STEPS],

    # Experiments multi matchup dataset fine tuning
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/multi_sat_preft.json"],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/multi_sat_preft.json",
    #  "--steps", *MODULES],
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/multi_sat_ft.json",
    #  "--steps", *MODELS_METS],

    # Experiments multi matchup dataset from scratch
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/multi_sat_scratch.json",
    #  "--steps", *MODELS_METS],

    #                                   DropTheMic / Pierre datasets

    # Experiments multi matchup dataset
    # [sys.executable, "./bin/split_data.py ", "--exp_config", "experiments_config/multi_sat_pierre.json"],

    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/multi_sat_ft_pierre.json",
    #  "--steps", *MODELS_METS],

    # Experiments multi matchup dataset from scratch
    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/multi_sat_scratch_pierre.json",
    #  "--steps", *MODELS_METS],

    # [sys.executable, "./bin/run_pipeline.py ", "--exp_config", "experiments_config/multi_sat_pierre.json",
    #  "--steps", *ALL],
]


for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
