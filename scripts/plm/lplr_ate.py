from montecover.plm import LPLRATECoverageSimulation

# Create and run simulation with config file
sim = LPLRATECoverageSimulation(
    config_file="scripts/plm/lplr_ate_config.yml",
    log_level="INFO",
    log_file="logs/plm/plr_ate_sim.log",
)
print("Calling file")
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="lplr_ate")

# Save config file for reproducibility
sim.save_config("results/plm/lplr_ate_config.yml")