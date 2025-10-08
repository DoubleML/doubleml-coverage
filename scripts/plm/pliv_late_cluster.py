from montecover.plm import PLIVLATEClusterCoverageSimulation

# Create and run simulation with config file
sim = PLIVLATEClusterCoverageSimulation(
    config_file="scripts/plm/pliv_late_cluster_config.yml",
    log_level="INFO",
    log_file="logs/plm/pliv_late_cluster_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="pliv_late_cluster")

# Save config file for reproducibility
sim.save_config("results/plm/pliv_late_cluster_config.yml")
