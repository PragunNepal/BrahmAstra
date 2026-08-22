import os
import re
import glob
import numpy as np
import pyvista as pv
from math import ceil
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

class CosmoVis:
    def __init__(self, data_dir="."):
        self.data_dir = Path(data_dir).resolve()

    def animate_hi_map(self, output_filename="HImap_Simulation.mp4",
                       rotate=True, rotation_speed=0.2, elev0=3, azim0=30,
                       pause_per_redshift_sec=0.0, freeze_end_sec=2.0):
        print("Initializing 4K PyVista HI Map Renderer...")
        
        vmax = 100.0
        frame_rate = 30
        window_size = (3840, 2160)
        frames_per_step = 30 
        
        ionz_dir = self.data_dir / "ionz_out"
        i_hi_map = str(ionz_dir / 'HI_map_*')
        i_pk = str(ionz_dir / 'pk.ionzs*_*')

        file_list = []
        for fn in glob.glob(i_hi_map):
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
                    mm = re.match(r'pk\.ionzs(\d+\.?\d*)_(\d+\.?\d*)$', os.path.basename(pf))
                    if mm and abs(float(mm.group(2)) - z) < 1e-6:
                        xhi = float(mm.group(1))
                        break
                if xhi is None: 
                    continue
                
                with open(fn, 'rb') as f:
                    mx, my, mz = np.fromfile(f, count=3, dtype=np.int32)
                    data = np.fromfile(f, count=mx * my * mz, dtype=np.float32)
                arr = data.reshape((mx, my, mz), order='C') * (Tb * xhi)
                arr = np.clip(arr, 0, vmax)
                frames.append((z, xhi, arr))
            return frames

        frames = load_frames()
        if not frames: 
            raise RuntimeError("No valid HI_map + pk.ionzs pairs found.")

        nx, ny, nz = frames[0][2].shape
        grid_template = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1), spacing=(1, 1, 1), origin=(0, 0, 0))

        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.open_movie(output_filename, framerate=frame_rate)
        plotter.set_background("black")

        grid = grid_template.copy()
        grid.cell_data['values'] = frames[0][2].flatten(order='F')
        vol = plotter.add_volume(grid, scalars='values', cmap='plasma', clim=[0, vmax], opacity='linear', shade=True, show_scalar_bar=False)

        plotter.add_scalar_bar(title=r'T(beta) x(HI) [mK]', title_font_size=30, label_font_size=26, vertical=False, position_x=0.25, position_y=0.02, width=0.5, height=0.04, color='white')

        plotter.camera_position = 'iso'
        plotter.camera.elevation = elev0
        plotter.camera.azimuth = azim0
        plotter.camera.zoom(0.6)

        pause_frames = int(frame_rate * pause_per_redshift_sec)
        freeze_end_frames = int(frame_rate * freeze_end_sec)
        total_frames = (len(frames) * pause_frames) + ((len(frames) - 1) * frames_per_step) + freeze_end_frames
        
        pbar = tqdm(total=total_frames, desc="Rendering PyVista Video", unit="frame")
        angle = 0

        print("Rendering Redshift Evolution...")
        for i in range(len(frames) - 1):
            z1, xhi1, arr1 = frames[i]
            z2, xhi2, arr2 = frames[i + 1]

            # 1. Hold on current redshift
            for _ in range(pause_frames):
                if rotate: 
                    angle += rotation_speed
                grid = grid_template.copy()
                grid.cell_data['values'] = arr1.flatten(order='F')
                plotter.remove_actor(vol)
                vol = plotter.add_volume(grid, scalars='values', cmap='plasma', clim=[0, vmax], opacity='linear', shade=True, show_scalar_bar=False)
                info_text = plotter.add_text(f'z={z1:.3f}\nxHI={xhi1:.3f}', position='lower_right', font_size=20, color='white')
                plotter.camera.azimuth = azim0 + angle
                plotter.camera.elevation = elev0
                plotter.render()
                plotter.write_frame()
                plotter.remove_actor(info_text)
                pbar.update(1)

            # 2. Interpolate to next redshift
            for j in range(frames_per_step):
                t = j / frames_per_step
                z_interp = (1 - t) * z1 + t * z2
                xhi_interp = (1 - t) * xhi1 + t * xhi2
                arr_interp = (1 - t) * arr1 + t * arr2

                if rotate: 
                    angle += rotation_speed
                grid = grid_template.copy()
                grid.cell_data['values'] = arr_interp.flatten(order='F')
                plotter.remove_actor(vol)
                vol = plotter.add_volume(grid, scalars='values', cmap='plasma', clim=[0, vmax], opacity='linear', shade=True, show_scalar_bar=False)
                info_text = plotter.add_text(f'z={z_interp:.3f}\nxHI={xhi_interp:.3f}', position='lower_right', font_size=20, color='white')
                plotter.camera.azimuth = azim0 + angle
                plotter.camera.elevation = elev0
                plotter.render()
                plotter.write_frame()
                plotter.remove_actor(info_text)
                pbar.update(1)

        # 3. Freeze at final redshift
        print("Rendering Final Freeze...")
        z_final, xhi_final, arr_final = frames[-1]
        for _ in range(pause_frames + freeze_end_frames):
            if rotate: 
                angle += rotation_speed
            grid = grid_template.copy()
            grid.cell_data['values'] = arr_final.flatten(order='F')
            plotter.remove_actor(vol)
            vol = plotter.add_volume(grid, scalars='values', cmap='plasma', clim=[0, vmax], opacity='linear', shade=True, show_scalar_bar=False)
            info_text = plotter.add_text(f'z={z_final:.3f}\nxHI={xhi_final:.3f}', position='lower_right', font_size=20, color='white')
            plotter.camera.azimuth = azim0 + angle
            plotter.camera.elevation = elev0
            plotter.render()
            plotter.write_frame()
            plotter.remove_actor(info_text)
            pbar.update(1)

        pbar.close()
        plotter.close()
        print(f"Animation saved as '{output_filename}'")

    def animate_hi_map_2d(self, output_filename="HImap_2D_Simulation.mp4",
                          pause_per_redshift_sec=0.0, freeze_end_sec=2.0):
        print("Initializing 2D Matplotlib HI Map Renderer (Fallback)...")
        
        vmax = 100.0
        fps = 30
        frames_per_step = 30
        pause_frames = int(fps * pause_per_redshift_sec)
        freeze_end_frames = int(fps * freeze_end_sec)
        
        ionz_dir = self.data_dir / "ionz_out"
        i_hi_map = str(ionz_dir / 'HI_map_*')
        i_pk = str(ionz_dir / 'pk.ionzs*_*')

        file_list = []
        for fn in glob.glob(i_hi_map):
            m = re.match(r'HI_map_(\d+\.?\d*)$', os.path.basename(fn))
            if m: 
                file_list.append((float(m.group(1)), fn))
        file_list.sort(reverse=True, key=lambda x: x[0])

        print("Loading 2D slices...")
        frames = []
        for z, fn in tqdm(file_list, desc="Loading data", unit="file"):
            Tb = 22.0 * np.sqrt((1 + z) / 7.0)
            xhi = None
            for pf in glob.glob(i_pk):
                mm = re.match(r'pk\.ionzs(\d+\.?\d*)_(\d+\.?\d*)$', os.path.basename(pf))
                if mm and abs(float(mm.group(2)) - z) < 1e-6:
                    xhi = float(mm.group(1))
                    break
            if xhi is None: 
                continue
            
            with open(fn, 'rb') as f:
                mx, my, mz = np.fromfile(f, count=3, dtype=np.int32)
                data = np.fromfile(f, count=mx * my * mz, dtype=np.float32)
            
            arr_3d = data.reshape((mx, my, mz), order='C') * (Tb * xhi)
            arr_slice = np.clip(arr_3d[:, :, mz // 2], 0, vmax)
            frames.append((z, xhi, arr_slice))

        if not frames: 
            raise RuntimeError("No valid HI_map pairs found.")

        frames_data, frames_z, frames_xhi = [], [], []
        for i in range(len(frames) - 1):
            z1, xhi1, arr1 = frames[i]
            z2, xhi2, arr2 = frames[i + 1]
            
            for _ in range(pause_frames):
                frames_data.append(arr1)
                frames_z.append(z1)
                frames_xhi.append(xhi1)
                
            for j in range(frames_per_step):
                t = j / frames_per_step
                arr_interp = (1 - t) * arr1 + t * arr2
                frames_data.append(arr_interp)
                frames_z.append((1 - t) * z1 + t * z2)
                frames_xhi.append((1 - t) * xhi1 + t * xhi2)

        for _ in range(pause_frames + freeze_end_frames):
            frames_data.append(frames[-1][2])
            frames_z.append(frames[-1][0])
            frames_xhi.append(frames[-1][1])

        total_frames = len(frames_data)
        
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.axis('off')
        
        cax = ax.imshow(frames_data[0], cmap='plasma', vmin=0, vmax=vmax, origin='lower')
        txt = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)
        
        def update(idx):
            cax.set_array(frames_data[idx])
            txt.set_text(f'z = {frames_z[idx]:.3f}\nxHI = {frames_xhi[idx]:.3f}')
            return cax, txt

        print(f"Starting 2D Matplotlib Render ({total_frames} frames)...")
        pbar = tqdm(total=total_frames, unit='frames', desc="Rendering MP4")
        ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
        writer = FFMpegWriter(fps=fps, bitrate=20000)
        ani.save(output_filename, writer=writer, progress_callback=lambda i, n: pbar.update(1))
        pbar.close()
        plt.close()
        print(f"2D Animation saved as '{output_filename}'")

    def animate_nbody(self, output_filename="BrahmAstra_Nbody.mp4", 
                      show_grid=False, grid_color='purple',
                      rotate=True, rotation_speed=0.1, elev0=25, azim0=30,
                      pause_per_redshift_sec=0.0, freeze_end_sec=3.0, 
                      zoom_in=True, zoom_factor=1.2, zoom_duration_sec=2.0):
        print("Initializing 3D N-body Animator...")
        fps = 30
        pause_frames = int(fps * pause_per_redshift_sec)
        extra_hold = int(fps * freeze_end_sec)
        zoom_frames = int(fps * zoom_duration_sec)
        
        search_pattern = str(self.data_dir / "output.nbody_*")
        files = []
        for fn in glob.glob(search_pattern):
            m = re.match(r'output\.nbody_(\d+\.?\d*)$', os.path.basename(fn))
            if m: 
                files.append((float(m.group(1)), fn))
        
        files.sort(reverse=True, key=lambda x: x[0])
        if not files: 
            raise RuntimeError("No N-body output files found.")
            
        redshifts = [z for z, fn in files]
        raw_frames = []
        
        print("Loading binary N-body data...")
        for z, fn in tqdm(files, desc="Loading N-body files", unit="file"):
            with open(fn, 'rb') as f:
                header = np.fromfile(f, dtype=np.int32, count=4)
                num_particles = header[3]
                f.seek(0)
                data = np.fromfile(f, dtype=np.float32)
                expected_floats = num_particles * 6
                particle_data = data[-expected_floats:]
                coords = particle_data.reshape(num_particles, 6)[:, :3]
                # FIREWALL: Strip out any NaN coordinates
                if np.isnan(coords).any():
                    coords = coords[~np.isnan(coords).any(axis=1)]
            raw_frames.append(coords)
            
        frames_data, frames_z = [], []
        for i in range(len(raw_frames) - 1):
            a, za = raw_frames[i], redshifts[i]
            b, zb = raw_frames[i+1], redshifts[i+1]
            for _ in range(pause_frames):
                frames_data.append(a)
                frames_z.append(za)
            frames_data.append(a)
            frames_z.append(za)
            frames_data.append((a + b)/2)
            frames_z.append((za + zb)/2)
            
        for _ in range(pause_frames):
            frames_data.append(raw_frames[-1])
            frames_z.append(redshifts[-1])
        frames_data.append(raw_frames[-1])
        frames_z.append(redshifts[-1])
        
        total_motion = len(frames_data)
        total_frames = total_motion + extra_hold
        
        sample_idx = np.linspace(0, len(raw_frames)-1, min(len(raw_frames),10), dtype=int)
        samples = np.vstack([raw_frames[i] for i in sample_idx])
        
        bounds = []
        for i in range(3):
            mn, mx = samples[:, i].min(), samples[:, i].max()
            if mx == mn: bounds.append((mn - 1, mx + 1))
            else: bounds.append((mn - 0.2*(mx-mn), mx + 0.2*(mx-mn)))
            
        centers = [0.5*(mn+mx) for mn, mx in bounds]
        
        dpi = 300
        fig = plt.figure(figsize=(6,6), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        if not show_grid:
            ax.axis('off')
        else:
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis._axinfo['grid']['color'] = grid_color
                axis._axinfo['grid']['linewidth'] = 0.1
                axis._axinfo['grid']['alpha'] = 0.3
                axis.pane.set_facecolor(grid_color)
                axis.pane.set_edgecolor(grid_color)
                axis.pane.set_alpha(0.05)
            ax.tick_params(colors=grid_color, which='both', labelsize=6)
            ax.axis('on')
            
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev0, azim0)
        
        scat = ax.scatter([], [], [], s=0.5, alpha=0.1, c='cyan', edgecolors='none')
        txt = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='white')
        ax_triad = inset_axes(ax, width="20%", height="20%", loc='lower left')
        ax_triad.axis('off')
        
        def rotation_matrix(ang_deg):
            theta = np.deg2rad(ang_deg)
            Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta),  np.cos(theta), 0], [0, 0, 1]])
            Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta),  np.cos(theta)]])
            return Rz @ Rx
            
        def update(idx):
            read_idx = idx if idx < total_motion else total_motion - 1
            data, zval = frames_data[read_idx], frames_z[read_idx]
            scat._offsets3d = (data[:,0], data[:,1], data[:,2])
            txt.set_text(f'z = {zval:.3f}')
            
            angle = (idx * rotation_speed) if rotate else 0
            ax.view_init(elev0 + angle, azim0 + angle)
            
            if zoom_in and idx >= total_motion:
                h = idx - total_motion
                f = 1 + (zoom_factor-1)*(h/zoom_frames) if h <= zoom_frames else zoom_factor
            else:
                f = 1   
                
            for i, (mn, mx) in enumerate(bounds):
                c = centers[i]; r = (mx-mn)/2
                low, high = c - r/f, c + r/f
                if i==0: ax.set_xlim(low, high)
                elif i==1: ax.set_ylim(low, high)
                else: ax.set_zlim(low, high)
                
            ax_triad.clear()
            ax_triad.axis('off')
            R = rotation_matrix(angle)
            for color, (label, vec) in zip(['red','green','blue'], {'X':np.array([1,0,0]), 'Y':np.array([0,1,0]), 'Z':np.array([0,0,1])}.items()):
                vrot = R @ vec
                end = np.array([0.5,0.5]) + 0.3 * vrot[:2]
                ax_triad.annotate('', xy=end, xytext=(0.5,0.5), arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))
                ax_triad.text(end[0], end[1], label, color=color, fontsize=8, alpha=0.6)
            return scat, txt

        print(f"Starting Matplotlib Render ({total_frames} frames)...")
        pbar = tqdm(total=total_frames, unit='frames', desc="Rendering MP4")
        ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
        writer = FFMpegWriter(fps=fps, metadata={'artist':'PragunNepal'}, bitrate=30000)
        ani.save(output_filename, writer=writer, progress_callback=lambda i, n: pbar.update(1))
        pbar.close()
        plt.close()
        print(f"Animation saved as '{output_filename}'")

    def animate_halos(self, output_filename="BrahmAstra_Halos.mp4",
                      show_grid=False, grid_color='purple',
                      rotate=True, rotation_speed=0.1, elev0=25, azim0=30,
                      pause_per_redshift_sec=0.0, freeze_end_sec=3.0, 
                      zoom_in=True, zoom_factor=1.2, zoom_duration_sec=2.0):
        print("Initializing 3D Halo Formation Animator...")
        
        fps = 30
        pause_frames = int(fps * pause_per_redshift_sec)
        extra_hold = int(fps * freeze_end_sec)
        zoom_frames = int(fps * zoom_duration_sec)
        
        nbody_pattern = str(self.data_dir / "output.nbody_*")
        nbody_files = []
        for fn in glob.glob(nbody_pattern):
            m = re.match(r'output\.nbody_(\d+\.?\d*)$', os.path.basename(fn))
            if m: 
                nbody_files.append((float(m.group(1)), fn))
        nbody_files.sort(reverse=True, key=lambda x: x[0])
        
        halo_files = []
        for fn in glob.glob(str(self.data_dir / "halo_*")):
            m = re.search(r'halo(?:_catalogue)?_(\d+\.?\d*)$', os.path.basename(fn))
            if m: 
                halo_files.append((float(m.group(1)), fn))
        halo_files.sort(reverse=True, key=lambda x: x[0])

        if not nbody_files: 
            raise RuntimeError("No N-body output files found.")

        print("Loading N-body binaries and Halo catalogs...")
        raw_n, z_n = [], []
        for z, fn in tqdm(nbody_files, desc="Loading N-body data", unit="file"):
            with open(fn, 'rb') as f:
                header = np.fromfile(f, dtype=np.int32, count=4)
                num_particles = header[3]
                f.seek(0)
                data = np.fromfile(f, dtype=np.float32)
                expected_floats = num_particles * 6
                particle_data = data[-expected_floats:]
                coords = particle_data.reshape(num_particles, 6)[:, :3]
                if np.isnan(coords).any():
                    coords = coords[~np.isnan(coords).any(axis=1)]
            raw_n.append(coords)
            z_n.append(z)

        raw_h = []
        z_h = [z for z, fn in halo_files]
        for z, fn in tqdm(halo_files, desc="Loading Halo catalogs", unit="file"):
            try:
                file_size = os.path.getsize(fn)
                if file_size == 0:
                    raw_h.append(np.zeros((0, 4)))
                    continue
                with open(fn, 'rb') as f:
                    first_int = np.fromfile(f, dtype=np.int32, count=1)[0]
                    f.seek(0)
                    data_f32 = np.fromfile(f, dtype=np.float32)

                if file_size == 4 + (first_int * 16): 
                    num_halos = first_int
                elif file_size >= 16 and (file_size - 16) % 16 == 0: 
                    num_halos = (file_size - 16) // 16
                else: 
                    num_halos = len(data_f32) // 4

                if num_halos > 0:
                    halo_data = data_f32[-(num_halos * 4):].reshape(num_halos, 4)
                    # FIREWALL: Strip out any NaN coordinates
                    if np.isnan(halo_data).any():
                        halo_data = halo_data[~np.isnan(halo_data).any(axis=1)]
                    raw_h.append(halo_data)
                else:
                    raw_h.append(np.zeros((0, 4)))
            except Exception:
                raw_h.append(np.zeros((0, 4)))

        matched_n, matched_h, matched_z = [], [], []
        for data_n, z in zip(raw_n, z_n):
            matched_n.append(data_n)
            matched_z.append(z)
            if z in z_h:
                idx = z_h.index(z)
                matched_h.append(raw_h[idx])
            else:
                all_z = np.array(z_h)
                if len(all_z) == 0: 
                    matched_h.append(np.zeros((0,4)))
                elif z > all_z[0]: 
                    matched_h.append(raw_h[0])
                elif z < all_z[-1]: 
                    matched_h.append(raw_h[-1])
                else:
                    i = np.where(all_z >= z)[0][-1]
                    j = i + 1
                    zA, zB = all_z[i], all_z[j]
                    A, B = raw_h[i], raw_h[j]
                    t = (z - zA) / (zB - zA) if zB != zA else 0
                    if A.size == 0 or B.size == 0:
                        interp = A if B.size == 0 else B
                    else:
                        tree = cKDTree(A[:, 1:])
                        _, idxs = tree.query(B[:, 1:], k=1)
                        A_matched = A[idxs]
                        interp = (1 - t) * A_matched + t * B
                    matched_h.append(interp)

        frames_n, frames_h, frames_z = [], [], []
        for i in range(len(matched_n) - 1):
            a_n, a_h, za = matched_n[i], matched_h[i], matched_z[i]
            b_n, b_h, zb = matched_n[i+1], matched_h[i+1], matched_z[i+1]
            
            for _ in range(pause_frames):
                frames_n.append(a_n)
                frames_h.append(a_h)
                frames_z.append(za)
                
            frames_n.append(a_n)
            frames_h.append(a_h)
            frames_z.append(za)
            frames_n.append((a_n + b_n)/2)
            
            if a_h.size == b_h.size and a_h.size > 0:
                frames_h.append((a_h + b_h)/2)
            else:
                frames_h.append(a_h)
            frames_z.append((za + zb)/2)

        for _ in range(pause_frames):
            frames_n.append(matched_n[-1])
            frames_h.append(matched_h[-1])
            frames_z.append(matched_z[-1])
        frames_n.append(matched_n[-1])
        frames_h.append(matched_h[-1])
        frames_z.append(matched_z[-1])

        total_motion = len(frames_n)
        total_frames = total_motion + extra_hold

        samp_n = np.vstack(frames_n[:min(10, total_motion)])
        valid_h = [f[:, 1:] for f in raw_h if f.size > 0]
        samp_h = np.vstack(valid_h) if valid_h else samp_n
        samples = np.vstack([samp_n, samp_h])
        
        # Fire-walling the bounding box calculation
        if np.isnan(samples).any():
            samples = samples[~np.isnan(samples).any(axis=1)]

        bounds = []
        for i in range(3):
            mn, mx = samples[:, i].min(), samples[:, i].max()
            if mx == mn: bounds.append((mn - 1, mx + 1))
            else: bounds.append((mn - 0.2*(mx-mn), mx + 0.2*(mx-mn)))
            
        centers = [0.5*(mn+mx) for mn, mx in bounds]

        dpi = 300
        fig = plt.figure(figsize=(6,6), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        if not show_grid:
            ax.axis('off')
        else:
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis._axinfo['grid']['color'] = grid_color
                axis._axinfo['grid']['linewidth'] = 0.1
                axis._axinfo['grid']['alpha'] = 0.3
                axis.pane.set_facecolor(grid_color)
                axis.pane.set_edgecolor(grid_color)
                axis.pane.set_alpha(0.05)
            ax.tick_params(colors=grid_color, which='both', labelsize=6)
            ax.axis('on')

        ax.set_box_aspect([1,1,1])
        ax.view_init(elev0, azim0)

        scn = ax.scatter([], [], [], s=0.5, alpha=0.1, c='cyan', edgecolors='none')
        sch = ax.scatter([], [], [], s=[], alpha=0.7, c='gold', edgecolors='none')
        txt = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='white')
        ax_triad = inset_axes(ax, width='20%', height='20%', loc='lower left')
        ax_triad.axis('off')

        def rotation_matrix(angle_deg):
            theta = np.deg2rad(angle_deg)
            Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta),  np.cos(theta), 0], [0, 0, 1]])
            Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta),  np.cos(theta)]])
            return Rz @ Rx

        def update(frame):
            idx = frame if frame < total_motion else total_motion - 1
            data_n, data_h, zval = frames_n[idx], frames_h[idx], frames_z[idx]

            scn._offsets3d = (data_n[:,0], data_n[:,1], data_n[:,2])
            if data_h.size:
                # FIREWALL: Prevent negative interpolated masses from throwing Matplotlib warnings
                radii = np.cbrt(np.maximum(data_h[:,0], 0))
                sizes = (radii / radii.max()) * 50 if radii.max() > 0 else radii * 0
                sch._offsets3d = (data_h[:,1], data_h[:,2], data_h[:,3])
                sch.set_sizes(np.maximum(sizes, 0))
            else:
                sch._offsets3d = ([], [], [])
                sch.set_sizes([])
                
            txt.set_text(f'z = {zval:.3f}')

            angle = (frame * rotation_speed) if rotate else 0
            ax.view_init(elev0 + angle, azim0 + angle)

            # FIX: Changed 'max_zoom' to the custom 'zoom_factor' parameter
            if zoom_in and frame >= total_motion:
                h = frame - total_motion
                f = 1 + (zoom_factor - 1) * (h / zoom_frames) if h <= zoom_frames else zoom_factor
            else:
                f = 1

            for i, (mn, mx) in enumerate(bounds):
                c = centers[i]; r = (mx - mn) / 2
                low, high = c - r/f, c + r/f
                if i == 0: ax.set_xlim(low, high)
                elif i == 1: ax.set_ylim(low, high)
                else: ax.set_zlim(low, high)

            ax_triad.clear()
            ax_triad.axis('off')
            R = rotation_matrix(angle)
            for color, (label, vec) in zip(['red','green','blue'], {'X':np.array([1,0,0]), 'Y':np.array([0,1,0]), 'Z':np.array([0,0,1])}.items()):
                v2 = R @ vec
                end = np.array([0.5, 0.5]) + 0.3 * v2[:2]
                ax_triad.annotate('', xy=end, xytext=(0.5, 0.5), arrowprops=dict(color=color, arrowstyle='->', alpha=0.6))
                ax_triad.text(end[0], end[1], label, color=color, fontsize=8, alpha=0.6)

            return scn, sch, txt

        print(f"Starting Matplotlib Render ({total_frames} frames)...")
        pbar = tqdm(total=total_frames, unit='frames', desc="Rendering MP4")
        ani = FuncAnimation(fig, update, frames=total_frames, interval=1000/fps, blit=False)
        writer = FFMpegWriter(fps=fps, metadata={'artist':'PragunNepal'}, bitrate=30000)
        ani.save(output_filename, writer=writer, progress_callback=lambda i, n: pbar.update(1))
        pbar.close()
        plt.close()
        print(f"Animation saved as '{output_filename}'")

    def plot_bispectrum(self, redshift=None, k_ratio=None, output_dir="bispec_plots"):
        import glob
        import re
        import os
        from pathlib import Path
        
        print("Initializing DviSukta Bispectrum Visualizer...")
        bispec_out_dir = self.data_dir / "bispec_out"
        out_path = self.data_dir / output_dir
        out_path.mkdir(exist_ok=True)
        
        if not bispec_out_dir.exists() or not list(bispec_out_dir.glob("z_*")):
            print("No bispec_out directory or redshift folders found. Did DviSukta run?")
            return

        # Sort redshifts from highest to lowest
        z_folders = sorted(glob.glob(str(bispec_out_dir / "z_*")), key=lambda x: float(os.path.basename(x).split('_')[1]), reverse=True)
        
        # 1. FILTER REDSHIFTS
        if redshift is not None:
            z_vals = [float(os.path.basename(f).split('_')[1]) for f in z_folders]
            closest_idx = np.argmin(np.abs(np.array(z_vals) - float(redshift)))
            target_z_dirs = [z_folders[closest_idx]]
        else:
            target_z_dirs = z_folders
            
        target_combinations = []
        
        # 2. FILTER K-RATIOS
        for z_dir in target_z_dirs:
            z_val = os.path.basename(z_dir).split('_')[1]
            k_folders = sorted(glob.glob(str(Path(z_dir) / "k2byk1_*")), key=lambda x: float(os.path.basename(x).split('_')[1]))
            
            if not k_folders:
                continue
                
            if k_ratio is not None:
                k_vals = [float(os.path.basename(f).split('_')[1]) for f in k_folders]
                closest_idx = np.argmin(np.abs(np.array(k_vals) - float(k_ratio)))
                target_k_dirs = [k_folders[closest_idx]]
            else:
                target_k_dirs = k_folders
                
            for k_dir in target_k_dirs:
                k_val = os.path.basename(k_dir).split('_')[1]
                target_combinations.append((z_dir, k_dir, z_val, k_val))
                
        if not target_combinations:
            print("No valid Bispectrum data combinations found to plot.")
            return

        print(f"Generating {len(target_combinations)} Bispectrum plots in '{output_dir}/'...")
        
        # 3. BATCH RENDER THE PLOTS
        for z_dir, k_dir, z_val, k_val in tqdm(target_combinations, desc="Rendering Maps", unit="plot"):
            files = glob.glob(f"{k_dir}/bispec_cosalpha*")
            if not files: 
                continue
                
            target_file = files[0]
            with open(target_file, 'rb') as f:
                data = np.fromfile(f, dtype=np.float32)
                
            match = re.search(r'c_data\d+\.\d+_(\d+)', os.path.basename(target_file))
            if match:
                n_elements = int(match.group(1))
                grid_size = int(np.sqrt(n_elements))
                heatmap_data = data[-n_elements:].reshape((grid_size, grid_size))
            else:
                side = int(np.sqrt(len(data)))
                heatmap_data = data[-(side**2):].reshape((side, side))
            
            mask = np.tril(np.ones(heatmap_data.shape), k=-1).astype(bool)
            heatmap_data[mask] = np.nan
            
            dpi = 300
            fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
            fig.patch.set_facecolor('white') 
            ax.set_facecolor('white')
            
            cax = ax.imshow(
                heatmap_data, 
                cmap='viridis',
                origin='lower', 
                aspect='auto',
                extent=[0.55, 0.95, 0.55, 0.95] 
            )
            ax.invert_yaxis()
            
            ax.set_title(f'DviSukta SABS: z={z_val}, $k_2/k_1$={k_val}', fontsize=12, pad=15)
            ax.set_xlabel(r'cos($\alpha$)', fontsize=10)
            ax.set_ylabel(r'$n = k_2/k_1$', fontsize=10)
            
            cbar = fig.colorbar(cax, ax=ax)
            cbar.set_label('Scaled Bispectrum', rotation=90, labelpad=15)

            # Save cleanly inside the new bispec_plots folder
            output_filename = out_path / f"Bispectrum_z{z_val}_k{k_val}.png"
            
            plt.tight_layout()
            plt.savefig(output_filename, facecolor=fig.get_facecolor(), edgecolor='none')
            
            # Critical: Close the figure in the loop so Jupyter doesn't run out of memory!
            plt.close(fig) 
            
        print(f"All {len(target_combinations)} plots saved successfully!")
