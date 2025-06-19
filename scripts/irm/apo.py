from montecover.irm import APOCoverageSimulation

# Create and run simulation with config file
sim = APOCoverageSimulation(
    config_file="scripts/irm/apo_config.yml",
    log_level="INFO",
    log_file="logs/irm/apo_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="apo")

# Save config file for reproducibility
sim.save_config("results/irm/apo_config.yml")
