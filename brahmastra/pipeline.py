from brahmastra.config import ConfigManager
from brahmastra.engine import NBodyRunner, FoFRunner, ReionYugaRunner, DviSuktaRunner

class BrahmAstraPipeline:
    def __init__(self, base_dir="."):
        self.config = ConfigManager(base_dir)
        self.nbody = NBodyRunner(base_dir)
        self.fof = FoFRunner(base_dir)
        self.reion = ReionYugaRunner(base_dir)
        self.dvisukta = DviSuktaRunner(base_dir)

    def run_full_simulation(self, grid_size=256, redshifts=[8.5], **kwargs):
        """
        Executes the entire BrahmAstra pipeline from start to finish.
        """
        print("==================================================")
        print("INITIALIZING BRAHMASTRA COSMOLOGY PIPELINE")
        print("==================================================\n")

        # 1. Write configurations dynamically
        print("-> Configuring C-Backends...")
        self.config.write_nbody_config(grid_size=grid_size, redshifts=redshifts, **kwargs)
        self.config.write_bispec_config() # Can pass kwargs here later if needed
        print("")

        # 2. Run N-body Engine (Dark Matter Grid)
        self.nbody.run()

        # 3. Run Friends-of-Friends (Halo Catalog)
        self.fof.run()

        # 4. Run ReionYuga (Ionization Maps)
        self.reion.run()

        # 5. Run DviSukta (Bispectrum Analyzer)
        self.dvisukta.run()

        print("\n==================================================")
        print("FULL SIMULATION PIPELINE COMPLETE")
        print("==================================================")
