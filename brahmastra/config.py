import numpy as np
from pathlib import Path

class ConfigManager:
    # 1. Accept BOTH custom_params and base_dir
    def __init__(self, custom_params=None, base_dir="."):
        self.base_dir = Path(base_dir).resolve()
        print("Initializing BrahmAstra Configuration...")
        
        self.params = {
            "grid_size": 64,       
            "box_size": 100.0,     
            "omega_m": 0.315,      
            "omega_l": 0.685,
            "hubble": 0.674,
            "sigma_8": 0.811,
            "n_s": 0.965,
            "z_start": 13.0,       
            "z_end": 7.0,          
            "num_redshifts": 4,    
            "redshifts": None,
            "n_bins": None      
        }

        if custom_params:
            for key, value in custom_params.items():
                if key in self.params:
                    self.params[key] = value
                else:
                    print(f"Warning: Unrecognized parameter '{key}' ignored.")

        if self.params["redshifts"] is None:
            self.params["redshifts"] = self.generate_redshifts()
        else:
            self.params["redshifts"] = np.array(self.params["redshifts"])
            
        print(f"   -> Grid Size: {self.params['grid_size']}^3")
        print(f"   -> Tracking {len(self.params['redshifts'])} redshifts: {np.round(self.params['redshifts'], 3)}")

    def generate_redshifts(self):
        z_start = self.params["z_start"]
        z_end = self.params["z_end"]
        num_z = self.params["num_redshifts"]
        scale_start = 1.0 + z_start
        scale_end = 1.0 + z_end
        z_array = np.geomspace(scale_start, scale_end, num=num_z) - 1.0
        return np.sort(z_array)[::-1]

    def get(self, key):
        return self.params.get(key)

    # ==========================================================
    # DO NOT DELETE YOUR FILE WRITERS BELOW THIS LINE!
    # (def write_nbody_config, def write_bispec_config, etc.)
    # ==========================================================

    def write_nbody_config(self, 
                           grid_size=256, 
                           redshifts=[8.5], 
                           nf=2, ll=0.07,
                           h=0.6704, omega_m=0.3183, omega_l=0.6817, n_s=0.9619,
                           omega_b=0.04902, sigma_8=0.8347,
                           seed=-100012, nbin=10, 
                           out_flag=0, pk_flag=1,
                           a_init=0.008, delta_a=0.004):
        
        noutput = len(redshifts)
        redshifts_str = " ".join(map(str, redshifts))
        
        content = f"""{seed}  {nbin}
{h}  {omega_m}  {omega_l}  {n_s} 
{omega_b}  {sigma_8}
{grid_size}  {grid_size}  {grid_size}  {nf}  {ll}
{out_flag}  {pk_flag}
{a_init}  {delta_a}
{noutput}
{redshifts_str}

#------ above are the parameter values (user may change this) --------
#----------------------------------------------------
seed  Nbin
hh    omega_m   omega_l     spectral_index
omega_baryon    sigma_8 
N1 N2 N3 fraction_fill LL
output_flag pk_flag
a_initial   delta_a
Noutput (# of redshifts where outputs are required)
List of the redshift values
#--------------------------------------------------------
"""
        # Write this configuration directly to the execution folder
        target_file = self.base_dir / "input.nbody_comp"
        with open(target_file, "w") as f:
            f.write(content)
        
        print(f"[ConfigManager] Successfully wrote N-body parameters for {noutput} redshift(s) to the root directory.")

    def write_bispec_config(self):
        box_size = self.get("box_size")
        grid_size = self.get("grid_size")
        custom_bins = self.get("n_bins")
        
        # If the user explicitly provided bin sizes, use them!
        if custom_bins is not None:
            safe_bins = int(custom_bins)
            print(f"[ConfigManager] Using custom DviSukta bins: {safe_bins}")
        else:
            # FIREWALL: Dynamic scaling based on Nyquist limit
            safe_bins = min(10, max(1, (grid_size // 2) - 2))
            print(f"[ConfigManager] Auto-scaling DviSukta bins to {safe_bins} (Grid={grid_size})")
            
        config_content = f"{box_size}\n{safe_bins}\n{safe_bins}\n{safe_bins}\n"
        
        with open(self.base_dir / "input.bispec", "w") as f:
            f.write(config_content)
