from montecover.irm import APOSCoverageSimulation

# Create and run simulation with config file
sim = APOSCoverageSimulation(
    config_file="scripts/irm/apos_config.yml",
    log_level="INFO",
    log_file="logs/irm/apos_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="apos")

# Save config file for reproducibility
sim.save_config("results/irm/apos_config.yml")
