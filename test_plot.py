from brahmastra.plot import CosmoVis

def main():
    print("--- BrahmAstra Visualization Test ---")
    
    # Initialize the visualizer
    vis = CosmoVis()
    
    # Render a 2D slice of the N-body density field generated in the root folder
    vis.render("output.nbody_8.500", dim='2D', slice_axis='y')

if __name__ == "__main__":
    main()
