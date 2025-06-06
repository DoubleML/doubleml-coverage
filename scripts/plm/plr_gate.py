from montecover.plm import PLRGATECoverageSimulation

# Create and run simulation with config file
sim = PLRGATECoverageSimulation(
    config_file="scripts/plm/plr_gate_config.yml",
    log_level="INFO",
    log_file="logs/plm/plr_gate_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="plr_gate")

# Save config file for reproducibility
sim.save_config("results/plm/plr_gate_config.yml")
