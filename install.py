import launch


if not launch.is_installed("wandb"):
    launch.run_pip("install wandb>=0.15.10", "Requirements for Weights & Biases")
