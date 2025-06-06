from montecover.irm import IRMCATECoverageSimulation

# Create and run simulation with config file
sim = IRMCATECoverageSimulation(
    config_file="scripts/irm/irm_cate_config.yml",
    log_level="INFO",
    log_file="logs/irm/irm_cate_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="irm_cate")

# Save config file for reproducibility
sim.save_config("results/irm/irm_cate_config.yml")
