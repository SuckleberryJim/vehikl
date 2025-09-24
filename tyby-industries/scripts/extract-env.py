import os
from pathlib import Path

fp = Path("vars.txt")
env = os.environ
targets = []

for x in env:
    i = x.lower()
    if "api" in i or "model" in i or "base" in i:
        targets.append(f"{x}={env[x]}")
else:
    fp.write_text("\n".join(targets))
