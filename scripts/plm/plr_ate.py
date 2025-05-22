
from montecover.plm import PLRATECoverageSimulation

# Create and run simulation with config file
sim = PLRATECoverageSimulation(
    config_file="scripts/plm/plr_ate_config.yml",
    log_level="INFO",
    log_file="logs/plm/plr_ate_sim.log"
)
sim.run_simulation()
sim.save_results(output_path="results/plm/", file_prefix="plr_ate")

# Save config file for reproducibility
sim.save_config("results/plm/plr_ate_config.yml")
