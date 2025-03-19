from montecover.did import DIDMultiCoverageSimulation

simulation = DIDMultiCoverageSimulation(
    repetitions=5,
    n_obs=500,
    output_path="results/did/did_pa_multi_coverage"
)
simulation.run_simulation()
simulation.save_results()