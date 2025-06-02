from montecover.plm import PLIVLATECoverageSimulation

# Create and run simulation with config file
sim = PLIVLATECoverageSimulation(
    config_file="scripts/plm/pliv_late_config.yml",
    log_level="INFO",
    log_file="logs/plm/pliv_late_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="pliv_late")

# Save config file for reproducibility
sim.save_config("results/plm/pliv_late_config.yml")
