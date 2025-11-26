from montecover.irm import APOSTuningCoverageSimulation

# Create and run simulation with config file
sim = APOSTuningCoverageSimulation(
    config_file="scripts/irm/apos_tune_config.yml",
    log_level="INFO",
    log_file="logs/irm/apos_tune_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="apos_tune")

# Save config file for reproducibility
sim.save_config("results/irm/apos_tune_config.yml")
