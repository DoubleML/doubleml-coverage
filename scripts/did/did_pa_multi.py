from montecover.did import DIDMultiCoverageSimulation

# Create and run simulation with config file
sim = DIDMultiCoverageSimulation(
    config_file="scripts/did/did_pa_multi_config.yml",
    log_level="DEBUG",
    log_file="logs/did/did_pa_multi_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/did/", file_prefix="did_pa_multi")

# Save config file for reproducibility
sim.save_config("results/did/did_pa_multi_config.yml")
