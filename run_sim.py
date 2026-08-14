from brahmastra.pipeline import BrahmAstraPipeline

def main():
    # Initialize the master pipeline
    sim = BrahmAstraPipeline()
    
    # Define a sequence of redshifts for the animation
    # We will use a smaller grid (64) so it runs quickly on your Mac
    z_sequence = [13.0, 11.0, 10.0, 9.0, 8.0, 7.0]
    
    # Run the full suite for all redshifts
    sim.run_full_simulation(grid_size=64, redshifts=z_sequence)

if __name__ == "__main__":
    main()
