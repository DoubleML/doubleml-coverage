from montecover.irm import IRMATECoverageSimulation

# Create and run simulation with config file
sim = IRMATECoverageSimulation(
    config_file="scripts/irm/irm_ate_config.yml",
    log_level="INFO",
    log_file="logs/irm/irm_ate_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="irm_ate")

# Save config file for reproducibility
sim.save_config("results/irm/irm_ate_config.yml")
