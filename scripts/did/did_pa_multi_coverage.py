
from montecover.did.did_multi import DIDMultiCoverageSimulation

# Create and run simulation with config file
sim = DIDMultiCoverageSimulation(
    config_file="scripts/did/multi_config.yml",
    log_level="DEBUG",
    log_file="logs/did/multi_sim.log"
)
sim.run_simulation()
sim.save_results(output_path="results/did/", file_prefix="did_multi")

# Save config file for reproducibility
sim.save_config("results/did/multi_config.yml")
