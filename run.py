"""Generate an AmodalExplorer world from an image and save renders and Gaussians."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "third_party" / "WonderWorld"
REVISION = "5cf11467ad21a9d7bca269f961855a235c7d352e"


def prepare_paths():
    if not (UPSTREAM / "models" / "models.py").exists():
        raise SystemExit("Run python scripts/prepare_wonderworld.py first.")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip()
    if head != REVISION:
        raise SystemExit(f"Expected WonderWorld {REVISION}, found {head}.")
    model_source = (UPSTREAM / "models" / "models.py").read_text()
    if "from amodalexplorer.lifting import refine_disparity" not in model_source:
        raise SystemExit("The WonderWorld checkout is missing patches/wonderworld.patch.")
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(UPSTREAM / "RepViT" / "sam"))
    os.chdir(ROOT)


def check_environment():
    import torch
    checks = {"cuda": torch.cuda.is_available(), "repvit_weights": (ROOT / "repvit_sam.pt").is_file()}
    for module in ("pytorch3d", "diff_gaussian_rasterization", "simple_knn", "repvit_sam",
                   "transformers", "diffusers", "kornia", "omegaconf", "spacy", "en_core_web_sm"):
        checks[module] = importlib.util.find_spec(module) is not None
    for module in ("pytorch3d._C", "diff_gaussian_rasterization._C", "simple_knn._C"):
        try:
            __import__(module)
            checks[module] = True
        except (ImportError, OSError, RuntimeError):
            checks[module] = False
    print(json.dumps(checks, indent=2))
    return all(checks.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/example.yaml")
    parser.add_argument("--output", help="override the output directory")
    parser.add_argument("--check", action="store_true", help="check dependencies without loading model weights")
    args = parser.parse_args()
    prepare_paths()
    if args.check:
        raise SystemExit(0 if check_environment() else 1)

    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from amodalexplorer.camera import step_camera

    if not torch.cuda.is_available():
        raise SystemExit("Generation requires a CUDA GPU. Use --check to inspect the environment.")
    from amodalexplorer.pipeline import WorldGenerator, load_models

    config = OmegaConf.merge(OmegaConf.load(UPSTREAM / "config" / "base-config.yaml"), OmegaConf.load(args.config))
    config.examples_yaml = str(Path(config.examples_yaml).resolve())
    config.sky_cache_dir = str(Path(config.sky_cache_dir).resolve())
    config.runs_dir = str(Path(args.output or config.runs_dir).resolve())
    output = Path(config.runs_dir)
    if output.exists():
        raise SystemExit(f"Output directory already exists: {output}. Choose a new --output.")
    output.mkdir(parents=True)
    OmegaConf.save(config, output / "config.yaml")
    generator = WorldGenerator(config, load_models(config))
    camera, info = generator.init_scene(config.example_name)
    scale = float(config.camera_step_relative) * info["median_depth"]

    def save_frame(index, cam, details):
        rgb = generator.render(cam)["rgb"]
        pixels = (rgb.detach().permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(pixels).save(output / f"frame{index:03d}.png")
        (output / f"frame{index:03d}.json").write_text(json.dumps({
            "R": cam.R[0].cpu().tolist(), "T": cam.T[0].cpu().tolist(), "info": details}, indent=2))

    save_frame(0, camera, info)
    for i, step in enumerate(config.camera_path, 1):
        camera = step_camera(camera, float(step.yaw), list(step.translation), scale)
        info = generator.generate_step(camera)
        save_frame(i, camera, info)
        print(f"Frame {i}: {info['calls']} hidden-layer completions")
    generator.gaussians.save_ply_all_with_filter(str(output / "world.ply"))
    print(f"Saved world.ply and frames to {output}")


if __name__ == "__main__":
    main()
