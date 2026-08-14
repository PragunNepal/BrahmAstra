import numpy as np

def inspect_header(filename):
    print(f"--- Inspecting Header of {filename} ---")
    try:
        with open(filename, "rb") as f:
            # Read the first 40 bytes (which is ten 32-bit values)
            raw_bytes = f.read(40)
            
            print("Raw Bytes:", raw_bytes)
            
            # Interpret those bytes as integers
            ints = np.frombuffer(raw_bytes, dtype=np.int32)
            print("\nInterpreted as int32:  ", ints)
            
            # Interpret those bytes as floats
            floats = np.frombuffer(raw_bytes, dtype=np.float32)
            print("Interpreted as float32:", floats)
            
    except FileNotFoundError:
        print(f"Could not find {filename}. Are you in the root directory?")

if __name__ == "__main__":
    # Looking at the file we generated earlier
    inspect_header("output.nbody_8.500")
