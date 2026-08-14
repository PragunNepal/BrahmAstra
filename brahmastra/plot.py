import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class CosmoVis:
    def __init__(self, base_dir="."):
        self.base_dir = Path(base_dir).resolve()

    def read_c_binary(self, filepath):
        """
        Reads the unformatted C binary output from BrahmAstra engines.
        Deciphered from Dr. Rajesh's reference script.
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Cannot find binary data at {file_path}")

        with open(file_path, 'rb') as f:
            # First 3 values are grid dimensions (Nx, Ny, Nz)
            dims = np.fromfile(f, count=3, dtype=np.int32)
            mesh_x, mesh_y, mesh_z = dims
            
            # The rest is the physical grid data (float32)
            expected_count = mesh_x * mesh_y * mesh_z
            data = np.fromfile(f, dtype=np.float32, count=expected_count)
            
        print(f"Loaded grid data with dimensions: {mesh_x}x{mesh_y}x{mesh_z}")
        return data.reshape((mesh_x, mesh_y, mesh_z), order='C')

    def render(self, filepath, dim='2D', slice_axis='y', slice_idx=None):
        """
        Master visualization router.
        """
        data = self.read_c_binary(filepath)
        
        if dim == '2D':
            self._plot_2d_slice(data, slice_axis, slice_idx)
        elif dim == '3D':
            self._plot_3d_volume(data)
        else:
            raise ValueError("dim must be '2D' or '3D'")

    def _plot_2d_slice(self, data, axis='y', idx=None):
        """
        Generates a 2D slice based on Dr. Rajesh's formatting logic.
        """
        shape = data.shape
        
        # Default to the middle slice if no index is provided
        if axis == 'x':
            idx = idx if idx is not None else shape[0] // 2
            slice_data = data[idx, :, :]
            title = f"2D Slice (X = {idx})"
        elif axis == 'y':
            idx = idx if idx is not None else shape[1] // 2
            slice_data = data[:, idx, :]
            title = f"2D Slice (Y = {idx})"
        else:
            idx = idx if idx is not None else shape[2] // 2
            slice_data = data[:, :, idx]
            title = f"2D Slice (Z = {idx})"

        plt.figure(figsize=(10, 8))
        im = plt.imshow(slice_data, cmap='viridis', origin='lower')
        cbar = plt.colorbar(im, pad=0.02)
        cbar.set_label("Grid Value", fontsize=12)
        plt.title(title, fontsize=14)
        plt.xlabel("Grid Cells")
        plt.ylabel("Grid Cells")
        plt.show()

    def _plot_3d_volume(self, data):
        """
        Placeholder for your custom 3D logic.
        """
        print("3D rendering initialized. Awaiting your Jupyter notebook logic!")
