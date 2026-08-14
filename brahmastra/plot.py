import os
import re
import glob
import numpy as np
import pyvista as pv
from math import ceil
from pathlib import Path

class CosmoVis:
    def __init__(self, data_dir="."):
        self.data_dir = Path(data_dir).resolve()

    def animate_hi_map(self, output_filename="HImap_Simulation.mp4"):
        print("🚀 Initializing 4K PyVista HI Map Renderer...")
        
        vmax = 100.0
        # Reduced durations slightly for this initial test so it renders quickly!
        initial_spin_duration_sec = 3 
        frame_rate = 30
        window_size = (3840, 2160)
        min_total_duration_sec = 10 
        
        ionz_dir = self.data_dir / "ionz_out"
        i_hi_map = str(ionz_dir / 'HI_map_*')
        i_pk = str(ionz_dir / 'pk.ionzs*_*')

        # --- START REPLACE ---
        file_list = []
        for fn in glob.glob(i_hi_map):
            # Using standard capture groups (...) to avoid < > symbol stripping
            m = re.match(r'HI_map_(\d+\.?\d*)$', os.path.basename(fn))
            if not m:
                continue
            z = float(m.group(1))
            file_list.append((z, fn))
        
        file_list.sort(reverse=True, key=lambda x: x[0])

        def load_frames():
            frames = []
            for z, fn in file_list:
                Tb = 22.0 * np.sqrt((1 + z) / 7.0)
                xhi = None
                for pf in glob.glob(i_pk):
                    # Standard capture groups for xhi (1) and z2 (2)
                    mm = re.match(r'pk\.ionzs(\d+\.?\d*)_(\d+\.?\d*)$', os.path.basename(pf))
                    if mm and abs(float(mm.group(2)) - z) < 1e-6:
                        xhi = float(mm.group(1))
                        break
                if xhi is None:
                    continue
                
                # Lightning-fast binary reader
                with open(fn, 'rb') as f:
                    mx, my, mz = np.fromfile(f, count=3, dtype=np.int32)
                    data = np.fromfile(f, count=mx * my * mz, dtype=np.float32)
                arr = data.reshape((mx, my, mz), order='C') * (Tb * xhi)
                arr = np.clip(arr, 0, vmax)
                frames.append((z, xhi, arr))
            return frames
        # --- END REPLACE ---

        frames = load_frames()
        if not frames:
            raise RuntimeError("No valid HI_map + pk.ionzs pairs found.")

        nx, ny, nz = frames[0][2].shape
        grid_template = pv.ImageData(
            dimensions=(nx + 1, ny + 1, nz + 1),
            spacing=(1, 1, 1),
            origin=(0, 0, 0)
        )

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.open_movie(output_filename, framerate=frame_rate)
        plotter.set_background("black")

        grid = grid_template.copy()
        grid.cell_data['values'] = frames[0][2].flatten(order='F')
        vol = plotter.add_volume(
            grid, scalars='values', cmap='plasma', clim=[0, vmax],
            opacity='linear', shade=True, show_scalar_bar=False
        )

        plotter.add_scalar_bar(
            title=r'T(beta) x(HI) [mK]', title_font_size=30, label_font_size=26,
            vertical=False, position_x=0.25, position_y=0.02,
            width=0.5, height=0.04, color='white'
        )

        azim0, elev0 = 30, 3
        plotter.camera_position = 'iso'
        plotter.camera.elevation = elev0
        plotter.camera.azimuth = azim0
        plotter.camera.zoom(0.6)

        N = len(frames)
        initial_spin_frames = initial_spin_duration_sec * frame_rate
        remaining_frames = max(min_total_duration_sec * frame_rate - initial_spin_frames, 1)
        frames_per_step = ceil(remaining_frames / (N - 1)) if N > 1 else 1
        rotation_speed = 0.2
        angle = 0

        print("Spinning initial frame...")
        for _ in range(initial_spin_frames):
            angle += rotation_speed
            grid = grid_template.copy()
            grid.cell_data['values'] = frames[0][2].flatten(order='F')
            plotter.remove_actor(vol)
            vol = plotter.add_volume(
                grid, scalars='values', cmap='plasma', clim=[0, vmax],
                opacity='linear', shade=True, show_scalar_bar=False
            )
            info_text = plotter.add_text(
                f'z={frames[0][0]:.3f}\nxHI={frames[0][1]:.3f}',
                position='lower_right', font_size=20, color='white'
            )
            plotter.camera.azimuth = azim0 + angle
            plotter.camera.elevation = elev0
            plotter.render()
            plotter.write_frame()
            plotter.remove_actor(info_text)

        print("Interpolating redshifts...")
        for i in range(len(frames) - 1):
            z1, xhi1, arr1 = frames[i]
            z2, xhi2, arr2 = frames[i + 1]

            for j in range(frames_per_step):
                t = j / frames_per_step
                z_interp = (1 - t) * z1 + t * z2
                xhi_interp = (1 - t) * xhi1 + t * xhi2
                arr_interp = (1 - t) * arr1 + t * arr2

                grid = grid_template.copy()
                grid.cell_data['values'] = arr_interp.flatten(order='F')
                plotter.remove_actor(vol)
                vol = plotter.add_volume(
                    grid, scalars='values', cmap='plasma', clim=[0, vmax],
                    opacity='linear', shade=True, show_scalar_bar=False
                )

                info_text = plotter.add_text(
                    f'z={z_interp:.3f}\nxHI={xhi_interp:.3f}',
                    position='lower_right', font_size=20, color='white'
                )

                angle += rotation_speed
                plotter.camera.azimuth = azim0 + angle
                plotter.camera.elevation = elev0
                plotter.render()
                plotter.write_frame()
                plotter.remove_actor(info_text)

        plotter.close()
        print(f"✅ Animation saved as '{output_filename}'")
