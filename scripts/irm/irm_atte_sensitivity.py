from montecover.irm import IRMATTESensitivityCoverageSimulation

# Create and run simulation with config file
sim = IRMATTESensitivityCoverageSimulation(
    config_file="scripts/irm/irm_atte_sensitivity_config.yml",
    log_level="INFO",
    log_file="logs/irm/irm_atte_sensitivity_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/irm/", file_prefix="irm_atte_sensitivity")

# Save config file for reproducibility
sim.save_config("results/irm/irm_atte_sensitivity_config.yml")
