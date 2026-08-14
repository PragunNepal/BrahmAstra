from brahmastra.plot import CosmoVis

def main():
    print("--- BrahmAstra Visualization Test ---")
    
    vis = CosmoVis()
    
    # Render a 2D slice of the Ionization map from the ionz_out directory!
    vis.render("ionz_out/HI_map_8.500", dim='2D', slice_axis='y')

if __name__ == "__main__":
    main()
