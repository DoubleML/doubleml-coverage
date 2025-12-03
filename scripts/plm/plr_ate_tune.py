from montecover.plm import PLRATETuningCoverageSimulation

# Create and run simulation with config file
sim = PLRATETuningCoverageSimulation(
    config_file="scripts/plm/plr_ate_tune_config.yml",
    log_level="INFO",
    log_file="logs/plm/plr_ate_tune_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="plr_ate_tune")

# Save config file for reproducibility
sim.save_config("results/plm/plr_ate_tune_config.yml")
