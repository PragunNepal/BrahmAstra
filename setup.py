from setuptools import setup, find_packages

setup(
    name="brahmastra",
    version="0.1.0",
    author="Pragun Nepal",
    description="A Python pipeline for Cosmological N-body, Halo, and HI simulations.",
    packages=find_packages(include=["brahmastra", "brahmastra.*"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "pyvista",
        "tqdm",
        "imageio",
        "imageio-ffmpeg"
    ],
    python_requires=">=3.8",
)
