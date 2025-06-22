from montecover.plm import LogisticATECoverageSimulation

# Create and run simulation with config file
sim = LogisticATECoverageSimulation(
    config_file="scripts/plm/logistic_ate_config.yml",
    log_level="INFO",
    log_file="logs/plm/logistic_ate_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="logistic_ate")

# Save config file for reproducibility
sim.save_config("results/plm/logistic_ate_config.yml")