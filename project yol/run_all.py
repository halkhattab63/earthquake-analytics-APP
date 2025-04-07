import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')


print("🚀 Running full pipeline...")

# Step 1: Run main.py
subprocess.run(["python", "main.py"], check=True)

# Step 2: Run pytest if available
try:
    subprocess.run(["pytest", "tests/test_integrity.py"], check=True)
except FileNotFoundError:
    print("⚠️ pytest not found. Run 'pip install pytest' to enable tests.")

print("✅ All done.")
