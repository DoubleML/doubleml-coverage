from montecover.irm import IRMATETuningCoverageSimulation

# Create and run simulation with config file
sim = IRMATETuningCoverageSimulation(
    config_file="scripts/irm/irm_ate_tune_config.yml",
    log_level="INFO",
    log_file="logs/irm/irm_ate_tune_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="irm_ate_tune")

# Save config file for reproducibility
sim.save_config("results/irm/irm_ate_tune_config.yml")
