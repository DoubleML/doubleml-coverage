from montecover.plm import LPLRATETuningCoverageSimulation

# Create and run simulation with config file
sim = LPLRATETuningCoverageSimulation(
    config_file="scripts/plm/lplr_ate_tune_config.yml",
    log_level="INFO",
    log_file="logs/plm/lplr_ate_tune_sim.log",
)
print("Calling file")
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="lplr_ate_tune")

# Save config file for reproducibility
sim.save_config("results/plm/lplr_ate_tune_config.yml")
