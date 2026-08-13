import subprocess
from pathlib import Path

class NBodyRunner:
    def __init__(self, executable_path="external/nbody/nbody_comp"):
        """
        Initializes the N-body execution engine.
        """
        self.base_dir = Path(__file__).parent.parent.resolve()
        self.exec_path = self.base_dir / executable_path
        
        if not self.exec_path.exists():
            raise FileNotFoundError(f"Executable not found at {self.exec_path}. Did you run 'make'?")

    def generate_param_file(self, params: dict = None, output_filename="input.nbody_comp"):
        """
        Generates the text parameter file strictly required by the N-body C code.
        If no params are provided, it uses the standard default values.
        """
        if params is None:
            params = {}

        param_path = self.base_dir / output_filename
        
        # Handle dynamic redshift list
        redshifts = params.get('redshifts', [8.5])
        n_output = len(redshifts)
        redshifts_str = " ".join(map(str, redshifts))

        # Build the exact string format expected by fscanf in C
        content = f"""{params.get('seed', -100012)}  {params.get('Nbin', 10)}
{params.get('hh', 0.6704)}  {params.get('omega_m', 0.3183)}  {params.get('omega_l', 0.6817)}  {params.get('spectral_index', 0.9619)}
{params.get('omega_baryon', 0.04902)}  {params.get('sigma_8', 0.8347)}
{params.get('N1', 256)}  {params.get('N2', 256)}  {params.get('N3', 256)}  {params.get('fraction_fill', 2)}  {params.get('LL', 0.07)}
{params.get('output_flag', 0)}  {params.get('pk_flag', 1)}
{params.get('a_initial', 0.008)}  {params.get('delta_a', 0.004)}
{n_output}
{redshifts_str}
"""
        with open(param_path, 'w') as f:
            f.write(content)
        
        print(f"Parameter file generated at: {param_path}")
        return param_path

    def run(self, params: dict = None, param_filename="input.nbody_comp"):
        """
        Generates the parameter file and executes the C binary.
        """
        param_path = self.generate_param_file(params, param_filename)
        
        print(f"Starting N-body Simulation using {self.exec_path}...")
        
        # Trigger the C executable via Python (removed check=True)
        result = subprocess.run(
            [str(self.exec_path)], 
            cwd=str(self.base_dir),
            capture_output=True,
            text=True
        )
        
        print("\n--- N-body Simulation Complete! ---")
        
        # Print standard output regardless of the garbage exit code
        if result.stdout:
            print("Output:\n", result.stdout)
            
        if result.stderr:
            print("\nError Details (if any):\n", result.stderr)
