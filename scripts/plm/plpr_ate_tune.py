from montecover.plm import PLPRATETuningCoverageSimulation

# Create and run simulation with config file
sim = PLPRATETuningCoverageSimulation(
    config_file="scripts/plm/plpr_ate_tune_config.yml",
    log_level="INFO",
    log_file="logs/plm/plpr_ate_tune_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="plpr_ate_tune")

# Save config file for reproducibility
sim.save_config("results/plm/plpr_ate_tune_config.yml")
