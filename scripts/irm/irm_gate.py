from montecover.irm import IRMGATECoverageSimulation

# Create and run simulation with config file
sim = IRMGATECoverageSimulation(
    config_file="scripts/irm/irm_gate_config.yml",
    log_level="INFO",
    log_file="logs/irm/irm_gate_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="irm_gate")

# Save config file for reproducibility
sim.save_config("results/irm/irm_gate_config.yml")
