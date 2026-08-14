from brahmastra.plot import CosmoVis

def main():
    vis = CosmoVis()
    # This will scan the ionz_out folder, read the binaries, and output the mp4!
    vis.animate_hi_map(output_filename="BrahmAstra_HImap.mp4")

if __name__ == "__main__":
    main()
