import subprocess
from pathlib import Path

# Bring your classes to the top level of the package
from .pipeline import BrahmAstraPipeline
from .plot import CosmoVis

def build_engines():
    """
    Finds all C-engines in the external/ directory and compiles them using make.
    Streams the output directly to the Jupyter notebook.
    """
    print("Initializing C-Compiler for BrahmAstra Engines...")
    
    # ... (Keep the rest of your build_engines function exactly as it is) ...
    
    # Dynamically find the external directory relative to this file
    base_dir = Path(__file__).resolve().parent.parent / "external"
    engines = ["nbody", "fof", "reionyuga", "dvisukta"]
    
    for engine in engines:
        engine_path = base_dir / engine
        if not engine_path.exists():
            print(f"Error: Could not find directory {engine_path}")
            continue
            
        print(f"Compiling {engine.upper()}...")
        
        try:
            # 1. Clean previous builds silently
            subprocess.run(["make", "clean"], cwd=engine_path, capture_output=True)
            
            # 2. Run Make and capture the output
            result = subprocess.run(["make"], cwd=engine_path, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"{engine.upper()} compiled successfully!")
            else:
                print(f"Failed to compile {engine.upper()}.\nError Log:\n{result.stderr}")
        except FileNotFoundError:
            print(f"System error: 'make' command not found. Do you have a C-compiler installed?")
        except Exception as e:
            print(f"Unexpected error compiling {engine.upper()}: {e}")
            
    print("All systems ready! You can now run your simulations.")
