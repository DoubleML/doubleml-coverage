from montecover.rdd import RDDCoverageSimulation

# Create and run simulation with config file
sim = RDDCoverageSimulation(
    config_file="scripts/rdd/rdd_fuzzy_config.yml",
    log_level="INFO",
    log_file="logs/rdd/rdd_fuzzy_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/rdd/", file_prefix="rdd_fuzzy")

# Save config file for reproducibility
sim.save_config("results/rdd/rdd_fuzzy_config.yml")
