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

class CosmoVis:
    def __init__(self, data_dir="."):
        self.data_dir = Path(data_dir).resolve()

    def animate_hi_map(self, output_filename="HImap_Simulation.mp4"):
        print("🚀 Initializing 4K PyVista HI Map Renderer...")
        
        vmax = 100.0
        initial_spin_duration_sec = 3 
        frame_rate = 30
        window_size = (3840, 2160)
        min_total_duration_sec = 10 
        
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

    def animate_nbody(self, with_axes=False, output_filename="BrahmAstra_Nbody.mp4"):
        print("🚀 Initializing 3D N-body Animator...")
        from tqdm import tqdm
        
        fps = 30
        rotation_speed = 0.1
        extra_hold = fps * 33  
        zoom_in, zoom_out = fps * 15, fps * 5
        max_zoom = 1.2
        
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
        for z, fn in files:
            with open(fn, 'rb') as f:
                header = np.fromfile(f, dtype=np.int32, count=4)
                num_particles = header[3]
                f.seek(0)
                data = np.fromfile(f, dtype=np.float32)
                expected_floats = num_particles * 6
                particle_data = data[-expected_floats:]
                coords = particle_data.reshape(num_particles, 6)[:, :3]
            raw_frames.append(coords)
            
        frames_data, frames_z = [], []
        for (a, za), (b, zb) in zip(zip(raw_frames[:-1], redshifts[:-1]), zip(raw_frames[1:], redshifts[1:])):
            frames_data.append(a); frames_z.append(za)
            frames_data.append((a + b)/2); frames_z.append((za + zb)/2)
        frames_data.append(raw_frames[-1]); frames_z.append(redshifts[-1])
        
        total_motion = len(frames_data)
        total_frames = total_motion + extra_hold
        
        sample_idx = np.linspace(0, len(raw_frames)-1, min(len(raw_frames),10), dtype=int)
        samples = np.vstack([raw_frames[i] for i in sample_idx])
        bounds = [(samples[:,i].min(), samples[:,i].max()) for i in range(3)]
        bounds = [(mn - 0.2*(mx-mn), mx + 0.2*(mx-mn)) for mn, mx in bounds]
        centers = [0.5*(mn+mx) for mn, mx in bounds]
        
        dpi = 300
        fig = plt.figure(figsize=(6,6), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black'); fig.patch.set_facecolor('black')
        
        if not with_axes:
            ax.axis('off')
        else:
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis._axinfo['grid']['color'] = 'purple'
                axis._axinfo['grid']['linewidth'] = 0.1
                axis._axinfo['grid']['alpha'] = 0.001
                axis.pane.set_facecolor('purple')
                axis.pane.set_edgecolor('purple')
                axis.pane.set_alpha(0.001)
            ax.tick_params(colors='purple', which='both', labelsize=6)
            ax.axis('on')
            
        ax.set_box_aspect([1,1,1])
        elev0, azim0 = 25, 30
        ax.view_init(elev0, azim0)
        
        scat = ax.scatter([], [], [], s=0.5, alpha=0.1, c='cyan', edgecolors='none')
        txt = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='white')
        
        ax_triad = inset_axes(ax, width="20%", height="20%", loc='lower left')
        ax_triad.axis('off')
        
        def rotation_matrix(ang_deg):
            theta = np.deg2rad(ang_deg)
            Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta),  np.cos(theta), 0],
                           [0, 0, 1]])
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(theta), -np.sin(theta)],
                           [0, np.sin(theta),  np.cos(theta)]])
            return Rz @ Rx
            
        def update(idx):
            if idx < total_motion:
                data, zval = frames_data[idx], frames_z[idx]
                scat._offsets3d = (data[:,0], data[:,1], data[:,2])
                txt.set_text(f'z = {zval:.3f}')
            
            angle = idx * rotation_speed
            ax.view_init(elev0 + angle, azim0 + angle)
            
            if idx >= total_motion:
                h = idx - total_motion
                f = 1 + (max_zoom-1)*(h/zoom_in) if h <= zoom_in else max_zoom - (max_zoom-1)*min((h - zoom_in)/zoom_out,1)
            else:
                f = 1   
                
            for i, (mn, mx) in enumerate(bounds):
                c = centers[i]; r = (mx-mn)/2
                low, high = c - r/f, c + r/f
                if i==0: ax.set_xlim(low, high)
                elif i==1: ax.set_ylim(low, high)
                else: ax.set_zlim(low, high)
                
            ax_triad.clear(); ax_triad.axis('off')
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
        print(f"✅ Animation saved as '{output_filename}'")

    def animate_halos(self, with_axes=False, output_filename="BrahmAstra_Halos.mp4"):
        print("🚀 Initializing 3D Halo Formation Animator...")
        from tqdm import tqdm
        from scipy.spatial import cKDTree
        
        fps = 30
        rotation_speed = 0.1
        extra_hold = fps * 33
        zoom_in, zoom_out = fps * 15, fps * 5
        max_zoom = 1.2
        
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
        raw_n = []
        z_n = []
        for z, fn in nbody_files:
            with open(fn, 'rb') as f:
                header = np.fromfile(f, dtype=np.int32, count=4)
                num_particles = header[3]
                f.seek(0)
                data = np.fromfile(f, dtype=np.float32)
                expected_floats = num_particles * 6
                particle_data = data[-expected_floats:]
                coords = particle_data.reshape(num_particles, 6)[:, :3]
            raw_n.append(coords)
            z_n.append(z)

        raw_h = []
        z_h = [z for z, fn in halo_files]
        for z, fn in halo_files:
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
                    raw_h.append(halo_data)
                else:
                    raw_h.append(np.zeros((0, 4)))
            except Exception as e:
                print(f"Warning: Could not read {fn}: {e}")
                raw_h.append(np.zeros((0, 4)))

        frames_n, frames_h, frames_z = [], [], []
        for data_n, z in zip(raw_n, z_n):
            frames_n.append(data_n)
            frames_z.append(z)
            
            if z in z_h:
                idx = z_h.index(z)
                frames_h.append(raw_h[idx])
            else:
                all_z = np.array(z_h)
                if len(all_z) == 0:
                    frames_h.append(np.zeros((0,4)))
                elif z > all_z[0]:
                    frames_h.append(raw_h[0])
                elif z < all_z[-1]:
                    frames_h.append(raw_h[-1])
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
                    frames_h.append(interp)

        total_motion = len(frames_n)
        total_frames = total_motion + extra_hold

        samp_n = np.vstack(frames_n[:min(10, total_motion)])
        samp_h = np.vstack([f[:, 1:] for f in raw_h if f.size > 0]) if any(f.size > 0 for f in raw_h) else samp_n
        samples = np.vstack([samp_n, samp_h])
        bounds = [(samples[:, i].min(), samples[:, i].max()) for i in range(3)]
        bounds = [(mn - 0.2*(mx-mn), mx + 0.2*(mx-mn)) for mn, mx in bounds]
        centers = [0.5*(mn+mx) for mn, mx in bounds]

        dpi = 300
        fig = plt.figure(figsize=(6,6), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black'); fig.patch.set_facecolor('black')
        
        if not with_axes:
            ax.axis('off')
        else:
            for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
                axis._axinfo['grid']['color'] = 'purple'
                axis._axinfo['grid']['linewidth'] = 0.1
                axis._axinfo['grid']['alpha'] = 0.001
                axis.pane.set_facecolor('purple')
                axis.pane.set_edgecolor('purple')
                axis.pane.set_alpha(0.001)
            ax.tick_params(colors='purple', which='both', labelsize=6)
            ax.axis('on')

        ax.set_box_aspect([1,1,1])
        elev0, azim0 = 25, 30
        ax.view_init(elev0, azim0)

        scn = ax.scatter([], [], [], s=0.5, alpha=0.1, c='cyan', edgecolors='none')
        sch = ax.scatter([], [], [], s=[], alpha=0.7, c='gold', edgecolors='none')
        txt = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='white')
        
        ax_triad = inset_axes(ax, width='20%', height='20%', loc='lower left')
        ax_triad.axis('off')

        def rotation_matrix(angle_deg):
            theta = np.deg2rad(angle_deg)
            Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta),  np.cos(theta), 0],
                           [0, 0, 1]])
            Rx = np.array([[1, 0, 0],
                           [0, np.cos(theta), -np.sin(theta)],
                           [0, np.sin(theta),  np.cos(theta)]])
            return Rz @ Rx

        def update(frame):
            idx = frame if frame < total_motion else total_motion - 1
            data_n = frames_n[idx]
            data_h = frames_h[idx]
            zval = frames_z[idx]

            if frame < total_motion:
                scn._offsets3d = (data_n[:,0], data_n[:,1], data_n[:,2])
                
                if data_h.size:
                    masses = data_h[:,0]
                    coords = data_h[:,1:]
                    radii = np.cbrt(masses)
                    sizes = (radii / radii.max()) * 50 if radii.max() > 0 else radii * 0
                    sch._offsets3d = (coords[:,0], coords[:,1], coords[:,2])
                    sch.set_sizes(sizes)
                else:
                    sch._offsets3d = ([], [], [])
                    sch.set_sizes([])
                    
                txt.set_text(f'z = {zval:.3f}')

            angle = frame * rotation_speed
            ax.view_init(elev0 + angle, azim0 + angle)

            if frame >= total_motion:
                h = frame - total_motion
                f = 1 + (max_zoom - 1) * (h / zoom_in) if h <= zoom_in else max_zoom - (max_zoom - 1) * min((h - zoom_in) / zoom_out, 1)
            else:
                f = 1

            for i, (mn, mx) in enumerate(bounds):
                c = centers[i]
                r = (mx - mn) / 2
                low, high = c - r/f, c + r/f
                if i == 0: ax.set_xlim(low, high)
                elif i == 1: ax.set_ylim(low, high)
                else: ax.set_zlim(low, high)

            ax_triad.clear(); ax_triad.axis('off')
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
        print(f"✅ Animation saved as '{output_filename}'")

    def plot_bispectrum(self, output_filename="BrahmAstra_Bispectrum.png"):
        print("🚀 Initializing DviSukta Bispectrum Visualizer...")
        
        search_pattern = str(self.data_dir / "external/dvisukta/c_data*")
        files = glob.glob(search_pattern)
        
        if not files:
            print("No DviSukta c_data files found.")
            return

        target_file = files[0]
        print(f"Reading binary data from {os.path.basename(target_file)}...")
        
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

        # --- NEW SABS PUBLICATION PLOT LOGIC ---
        
        # 1. Mask out the physically impossible triangle configurations (lower triangle)
        mask = np.tril(np.ones(heatmap_data.shape), k=-1).astype(bool)
        heatmap_data[mask] = np.nan
        
        # 2. Setup the Matplotlib heatmap with a white background
        dpi = 300
        fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
        fig.patch.set_facecolor('white') 
        ax.set_facecolor('white')
        
        # 3. Add the physical extents for cos(alpha) and n=k2/k1
        cax = ax.imshow(
            heatmap_data, 
            cmap='viridis',
            origin='lower', 
            aspect='auto',
            extent=[0.55, 0.95, 0.55, 0.95] 
        )
        
        # 4. Styling to match the paper format
        ax.set_title(f'DviSukta SABS: {os.path.basename(target_file)}', fontsize=12, pad=15)
        ax.set_xlabel(r'cos($\alpha$)', fontsize=10)
        ax.set_ylabel(r'$n = k_2/k_1$', fontsize=10)
        
        # 5. Add the colorbar
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Scaled Bispectrum', rotation=90, labelpad=15)

        plt.tight_layout()
        plt.savefig(output_filename, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"✅ Bispectrum SABS heatmap saved as '{output_filename}'")
