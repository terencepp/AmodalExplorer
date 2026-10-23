"""Fetch a pinned WonderWorld checkout and apply the AmodalExplorer integration."""
import argparse
import subprocess
from pathlib import Path

REVISION = "5cf11467ad21a9d7bca269f961855a235c7d352e"
UPSTREAM = "https://github.com/KovenYu/WonderWorld.git"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="clone a local WonderWorld checkout instead of downloading")
    args = parser.parse_args()
    target = ROOT / "third_party" / "WonderWorld"
    if target.exists():
        parser.error(f"{target} already exists; use the prepared checkout or choose a fresh release directory")
    source = str(Path(args.source).resolve()) if args.source else UPSTREAM
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-hardlinks", "--no-checkout", "--", source, str(target)], check=True)
    subprocess.run(["git", "checkout", "--detach", REVISION], cwd=target, check=True)
    subprocess.run(["git", "remote", "set-url", "origin", UPSTREAM], cwd=target, check=True)
    patch = str(ROOT / "patches" / "wonderworld.patch")
    subprocess.run(["git", "apply", "--check", patch], cwd=target, check=True)
    subprocess.run(["git", "apply", patch], cwd=target, check=True)
    print(f"Prepared WonderWorld {REVISION} in {target}")


if __name__ == "__main__":
    main()
