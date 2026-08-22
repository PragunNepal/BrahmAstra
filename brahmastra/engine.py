import subprocess
from pathlib import Path

class NBodyRunner:
    def __init__(self, base_dir="."):
        import os
        from pathlib import Path
        self.base_dir = Path(base_dir).resolve()
        self.exec_path = self.base_dir / "external" / "nbody" / "nbody_comp"
        
        if not self.exec_path.exists():
            raise FileNotFoundError(f"Executable not found at {self.exec_path}. Did you run 'make' in external/nbody?")

    def run(self):
        import subprocess
        print(f"Starting N-body engine using {self.exec_path}...")
        
        # Trigger the C executable
        result = subprocess.run(
            [str(self.exec_path)], 
            cwd=str(self.base_dir),
            capture_output=True,
            text=True
        )
        
        print("\n--- N-body Engine Complete! ---")
        
        if result.stdout:
            print("Output:\n", result.stdout)
            
        if result.stderr:
            print("\nError Details (if any):\n", result.stderr)
            
            
            
class FoFRunner:
    def __init__(self, base_dir="."):
        import os
        from pathlib import Path
        self.base_dir = Path(base_dir).resolve()
        self.exec_path = self.base_dir / "external" / "fof" / "fof_main"
        
        if not self.exec_path.exists():
            raise FileNotFoundError(f"Executable not found at {self.exec_path}. Did you run 'make' in external/fof?")

    def run(self):
        import subprocess
        print(f"Starting FoF Halo Finder using {self.exec_path}...")
        
        # Trigger the C executable via Python (ignoring garbage exit codes like before)
        result = subprocess.run(
            [str(self.exec_path)], 
            cwd=str(self.base_dir),
            capture_output=True,
            text=True
        )
        
        print("\n--- FoF Halo Finder Complete! ---")
        
        if result.stdout:
            print("Output:\n", result.stdout)
            
        if result.stderr:
            print("\nError Details (if any):\n", result.stderr)
            
            
            
            
class ReionYugaRunner:
    def __init__(self, base_dir="."):
        import os
        from pathlib import Path
        self.base_dir = Path(base_dir).resolve()
        self.exec_path = self.base_dir / "external" / "reionyuga" / "ionz_main"
        
        if not self.exec_path.exists():
            raise FileNotFoundError(f"Executable not found at {self.exec_path}. Did you run 'make' in external/reionyuga?")

    def run(self):
        import subprocess
        print(f"Starting ReionYuga Ionization engine using {self.exec_path}...")
        
        result = subprocess.run(
            [str(self.exec_path)], 
            cwd=str(self.base_dir),
            capture_output=True,
            text=True
        )
        
        print("\n--- ReionYuga Engine Complete! ---")
        
        if result.stdout:
            print("Output:\n", result.stdout)
            
        if result.stderr:
            print("\nError Details (if any):\n", result.stderr)
            
            
            
            
            
            
class DviSuktaRunner:
    def __init__(self, base_dir="."):
        import os
        from pathlib import Path
        self.base_dir = Path(base_dir).resolve()
        self.exec_path = self.base_dir / "external" / "dvisukta" / "bispec"
        
        if not self.exec_path.exists():
            raise FileNotFoundError(f"Executable not found at {self.exec_path}.")

    def run(self):
        import subprocess
        import glob
        import shutil
        import os
        import re
        from tqdm.auto import tqdm

        # Grab all ReionYuga HI_maps
        hi_maps = sorted(glob.glob(str(self.base_dir / "ionz_out" / "HI_maprs_*")))
        if not hi_maps:
            print("Error: No HI_maprs files found! ReionYuga must run first.")
            return
            
        # Create a master directory for all bispectrum outputs
        bispec_out_dir = self.base_dir / "bispec_out"
        bispec_out_dir.mkdir(exist_ok=True)
        
        ghost_file = self.base_dir / "c_data8.0_100"

        # Loop through every redshift map
        for target_map in tqdm(hi_maps, desc="Running DviSukta", unit="map"):
            
            # 1. Silently trick the C-engine with the Ghost File
            shutil.copy(target_map, ghost_file)

            # 2. Run the C-engine
            subprocess.run([str(self.exec_path)], cwd=str(self.base_dir), capture_output=True)
            
            # 3. Extract the redshift from the filename
            match = re.search(r'HI_maprs_(\d+\.?\d*)', os.path.basename(target_map))
            z_val = match.group(1) if match else "unknown"
            
            # 4. Move generated k2byk1 folders into a redshift-specific folder
            z_dir = bispec_out_dir / f"z_{z_val}"
            z_dir.mkdir(exist_ok=True)
            
            for k_folder in glob.glob(str(self.base_dir / "k2byk1_*")):
                folder_name = os.path.basename(k_folder)
                dest = z_dir / folder_name
                if dest.exists():
                    shutil.rmtree(dest) # Remove if it already exists from a previous run
                shutil.move(k_folder, z_dir)
                
        # Clean up the ghost file
        if ghost_file.exists():
            os.remove(ghost_file)
