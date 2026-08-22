from brahmastra.config import ConfigManager
from brahmastra.engine import NBodyRunner, FoFRunner, ReionYugaRunner, DviSuktaRunner
from tqdm.auto import tqdm  # <-- NEW: Import the progress bar
import sys

class BrahmAstraPipeline:
    def __init__(self, params=None, base_dir="."):
        self.config = ConfigManager(custom_params=params, base_dir=base_dir)
        self.nbody = NBodyRunner(base_dir)
        self.fof = FoFRunner(base_dir)
        self.reion = ReionYugaRunner(base_dir)
        self.dvisukta = DviSuktaRunner(base_dir)

    def run_full_simulation(self):
        print("==================================================")
        print(" INITIALIZING BRAHMASTRA COSMOLOGY PIPELINE")
        print("==================================================\n")

        print("-> Configuring C-Backends...")
        grid_size = self.config.get("grid_size")
        redshifts = self.config.get("redshifts")
        
        self.config.write_nbody_config(grid_size=grid_size, redshifts=redshifts)
        self.config.write_bispec_config() 

        # Define the 4 engines in order of their physics dependency
        engines = [
            ("N-Body (Dark Matter)", self.nbody),
            ("FoF (Halo Catalog)", self.fof),
            ("ReionYuga (HI Maps)", self.reion),
            ("DviSukta (Bispectrum)", self.dvisukta)
        ]

        print("\n-> Launching Compute Phase...")
        
        # Open a log file to hide the messy C-output
        log_path = self.config.base_dir / "simulation.log"
        with open(log_path, "w") as log_file:
            
            # The sleek Jupyter Progress Bar!
            for name, engine in tqdm(engines, desc="Pipeline Progress", unit="engine"):
                log_file.write(f"\n[{name}] STARTING...\n")
                
                # Temporarily redirect Python's print statements to the log file
                original_stdout = sys.stdout
                sys.stdout = log_file
                
                try:
                    engine.run()
                finally:
                    # Restore printing to the Jupyter notebook
                    sys.stdout = original_stdout

        print("\n==================================================")
        print("FULL SIMULATION PIPELINE COMPLETE")
        print(f"Detailed logs saved to: {log_path.name}")
        print("==================================================")
