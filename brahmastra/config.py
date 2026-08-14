import os
from pathlib import Path

class ConfigManager:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()

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

    def write_bispec_config(self, ll=1.12, nk1bin=10, nnbin=10, ncostbin=10):
        content = f"{ll}\n{nk1bin}\n{nnbin}\n{ncostbin}\n"
        target_file = self.base_dir / "input.bispec"
        
        with open(target_file, "w") as f:
            f.write(content)
        
        print("[ConfigManager] Successfully wrote DviSukta bispectrum parameters to the root directory.")
