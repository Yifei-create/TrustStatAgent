from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "zero_shot_cot": "https://github.com/kojima-takeshi188/zero_shot_cot.git",
    "react": "https://github.com/ysymyth/ReAct.git",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    target_root = ROOT / "third_party"
    provenance = {}
    for name, url in SOURCES.items():
        target = target_root / name
        if target.exists() and not args.refresh:
            raise FileExistsError(f"Refusing to overwrite existing checkout: {target}")
        if target.exists():
            raise FileExistsError(f"Remove or archive checkout before refresh: {target}")
        subprocess.run(["git", "clone", url, str(target)], check=True)
        revision = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        provenance[name] = {"url": url, "revision": revision}
    (target_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(target_root / "provenance.json")


if __name__ == "__main__":
    main()
