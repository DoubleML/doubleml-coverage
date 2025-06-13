from montecover.irm import LPQCoverageSimulation

# Create and run simulation with config file
sim = LPQCoverageSimulation(
    config_file="scripts/irm/lpq_config.yml",
    log_level="INFO",
    log_file="logs/irm/lpq_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="lpq")

# Save config file for reproducibility
sim.save_config("results/irm/lpq_config.yml")
