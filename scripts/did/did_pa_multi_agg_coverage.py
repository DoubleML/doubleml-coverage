from montecover.did import DIDMultiCoverageSimulation

simulation = DIDMultiCoverageSimulation(
    repetitions=5,
    n_obs=500,
)
simulation.run_simulation()
simulation.save_results(output_path="results/did/", file_prefix="did_pa_multi_coverage")