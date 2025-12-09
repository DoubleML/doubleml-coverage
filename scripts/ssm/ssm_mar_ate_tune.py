from montecover.ssm import SSMMarATETuningCoverageSimulation

# Create and run simulation with config file
sim = SSMMarATETuningCoverageSimulation(
    config_file="scripts/ssm/ssm_mar_ate_tune_config.yml",
    log_level="INFO",
    log_file="logs/ssm/ssm_mar_ate_tune_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/ssm/", file_prefix="ssm_mar_ate_tune")

# Save config file for reproducibility
sim.save_config("results/ssm/ssm_mar_ate_tune_config.yml")
