from brahmastra.engine import ReionYugaRunner

def main():
    print("--- BrahmAstra ReionYuga Interface Test ---")
    try:
        runner = ReionYugaRunner()
        runner.run()
    except Exception as e:
        print(f"\n[ERROR] ReionYuga execution failed: {e}")

if __name__ == "__main__":
    main()
