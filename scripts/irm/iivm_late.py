from montecover.irm import IIVMLATECoverageSimulation

# Create and run simulation with config file
sim = IIVMLATECoverageSimulation(
    config_file="scripts/irm/iivm_late_config.yml",
    log_level="INFO",
    log_file="logs/irm/iivm_late_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="iivm_late")

# Save config file for reproducibility
sim.save_config("results/irm/iivm_late_config.yml")
