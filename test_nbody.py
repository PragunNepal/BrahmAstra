from brahmastra.engine import NBodyRunner

def main():
    print("--- BrahmAstra Python Interface Test ---")
    
    # Initialize the runner
    runner = NBodyRunner()
    
    # Define a lightweight test parameter set for a fast run
    test_params = {
        'N1': 64,
        'N2': 64,
        'N3': 64,
        'redshifts': [8.5]
    }
    
    # Trigger the C binary through Python
    runner.run(params=test_params)

if __name__ == "__main__":
    main()
