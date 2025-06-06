from montecover.irm import CVARCoverageSimulation

# Create and run simulation with config file
sim = CVARCoverageSimulation(
    config_file="scripts/irm/cvar_config.yml",
    log_level="INFO",
    log_file="logs/irm/cvar_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="cvar")

# Save config file for reproducibility
sim.save_config("results/irm/cvar_config.yml")
