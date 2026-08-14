from brahmastra.pipeline import BrahmAstraPipeline

# Initialize and run a fast 64^3 test simulation at redshift 8.5
sim = BrahmAstraPipeline()
sim.run_full_simulation(grid_size=64, redshifts=[8.5])
