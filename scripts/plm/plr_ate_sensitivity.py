from montecover.plm import PLRATESensitivityCoverageSimulation

# Create and run simulation with config file
sim = PLRATESensitivityCoverageSimulation(
    config_file="scripts/plm/plr_ate_sensitivity_config.yml",
    log_level="INFO",
    log_file="logs/plm/plr_ate_sensitivity_sim.log",
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="plr_ate_sensitivity")

# Save config file for reproducibility
sim.save_config("results/plm/plr_ate_sensitivity_config.yml")
