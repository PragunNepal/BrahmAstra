from brahmastra.engine import FoFRunner

def main():
    print("--- BrahmAstra FoF Interface Test ---")
    try:
        runner = FoFRunner()
        runner.run()
    except Exception as e:
        print(f"\n[ERROR] FoF execution failed: {e}")

if __name__ == "__main__":
    main()
