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
            raise FileNotFoundError(f"Executable not found at {self.exec_path}. Did you run 'make' in external/dvisukta?")

    def run(self):
        import subprocess
        print(f"Starting DviSukta Bispectrum engine using {self.exec_path}...")
        
        result = subprocess.run(
            [str(self.exec_path)], 
            cwd=str(self.base_dir),
            capture_output=True,
            text=True
        )
        
        print("\n--- DviSukta Engine Complete! ---")
        
        if result.stdout:
            print("Output:\n", result.stdout)
            
        if result.stderr:
            print("\nError Details (if any):\n", result.stderr)
