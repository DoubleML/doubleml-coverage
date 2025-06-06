from montecover.ssm import SSMNonIgnorableATECoverageSimulation

# Create and run simulation with config file
sim = SSMNonIgnorableATECoverageSimulation(
    config_file="scripts/ssm/ssm_nonig_ate_config.yml",
    log_level="INFO",
    log_file="logs/ssm/ssm_nonig_ate_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/ssm/", file_prefix="ssm_nonig_ate")

# Save config file for reproducibility
sim.save_config("results/ssm/ssm_nonig_ate_config.yml")
