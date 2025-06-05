from montecover.irm import PQCoverageSimulation

# Create and run simulation with config file
sim = PQCoverageSimulation(
    config_file="scripts/irm/pq_config.yml",
    log_level="INFO",
    log_file="logs/irm/pq_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="pq")

# Save config file for reproducibility
sim.save_config("results/irm/pq_config.yml")
