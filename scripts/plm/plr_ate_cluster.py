from montecover.plm import PLRATEClusterCoverageSimulation

# Create and run simulation with config file
sim = PLRATEClusterCoverageSimulation(
    config_file="scripts/plm/plr_ate_cluster_config.yml",
    log_level="INFO",
    log_file="logs/plm/plr_ate_cluster_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="plr_ate_cluster")

# Save config file for reproducibility
sim.save_config("results/plm/plr_ate_cluster_config.yml")
