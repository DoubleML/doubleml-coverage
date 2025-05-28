from montecover.plm import PLRCATECoverageSimulation

# Create and run simulation with config file
sim = PLRCATECoverageSimulation(
    config_file="scripts/plm/plr_cate_config.yml",
    log_level="INFO",
    log_file="logs/plm/plr_cate_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="plr_cate")

# Save config file for reproducibility
sim.save_config("results/plm/plr_cate_config.yml")
