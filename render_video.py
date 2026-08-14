from brahmastra.plot import CosmoVis

def main():
    vis = CosmoVis()
    # Trigger the DviSukta 2D Heatmap!
    vis.plot_bispectrum(output_filename="BrahmAstra_Bispectrum.png")

if __name__ == "__main__":
    main()
