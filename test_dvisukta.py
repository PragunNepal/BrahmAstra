from brahmastra.engine import DviSuktaRunner

def main():
    print("--- BrahmAstra DviSukta Interface Test ---")
    try:
        runner = DviSuktaRunner()
        runner.run()
    except Exception as e:
        print(f"\n[ERROR] DviSukta execution failed: {e}")

if __name__ == "__main__":
    main()
