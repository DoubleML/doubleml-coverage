from montecover.plm import PLPRATECoverageSimulation

# Create and run simulation with config file
sim = PLPRATECoverageSimulation(
    config_file="scripts/plm/plpr_ate_config.yml",
    log_level="INFO",
    log_file="logs/plm/plpr_ate_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="plpr_ate")

# Save config file for reproducibility
sim.save_config("results/plm/plpr_ate_config.yml")
