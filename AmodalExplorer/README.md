# AmodalExplorer

**Training-Free Online 3D World Generation via Occlusion-Ordered Completion**

AmodalExplorer expands a single image into a persistent Gaussian world. At each keyframe, occlusion-ordered amodal peeling (OAP) recursively completes hidden content behind visible occluders. Geometry-aware lifting (GAL) fits depth-oriented segment planes before enforcing the depth-order constraint and adding the layers to the world.

This release contains the core generation method and a minimal image-to-world demo. It uses the frozen priors and Gaussian renderer of [WonderWorld](https://github.com/KovenYu/WonderWorld), pinned to commit `5cf11467ad21a9d7bca269f961855a235c7d352e`. Benchmark datasets, evaluation code, ablations, and experimental outputs are excluded.

## Installation

Use Linux, Python 3.10, a CUDA-compatible NVIDIA GPU, and approximately 48 GB VRAM, as in WonderWorld. CUDA is also needed to build the native extensions. The commands below follow WonderWorld's CUDA 12.4 / PyTorch 2.4 environment.

From this repository's root:

```bash
conda create -n amodalexplorer python=3.10 -y
conda activate amodalexplorer
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath

# Fetch the fixed upstream revision and apply the small integration patch.
python scripts/prepare_wonderworld.py

pip install -r requirements.txt
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
pip install third_party/WonderWorld/submodules/depth-diff-gaussian-rasterization-min
pip install third_party/WonderWorld/submodules/simple-knn
pip install -e third_party/WonderWorld/RepViT/sam
python -m spacy download en_core_web_sm

wget -O repvit_sam.pt https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_sam.pt
python run.py --check
```

The preparation script can instead clone an existing WonderWorld checkout with `--source /path/to/WonderWorld`. It reads that checkout's committed history and applies the patch only to the new copy. The destination must not already exist.

Model weights are downloaded from Hugging Face on the first generation run. The config uses the `sd2-community/stable-diffusion-2-inpainting` checkpoint mirror used by the research implementation; a compatible local checkpoint directory can also be supplied. No OpenAI API key is required.

## Generate a world

```bash
python run.py --config config/example.yaml
```

The example uses WonderWorld's campus image. The first run generates and caches a sky panorama; this can take substantially longer than later runs.

The output directory contains `world.ply`, rendered frames, camera poses and generation timing, plus WonderWorld's intermediate generation files. Use a new `--output output/my_run` for another run. The entry point refuses to overwrite an existing output directory.

Add your own image and caption to `config/examples.yaml`, then set `example_name` in the run config. Images are resized to 512 × 512; supply a focal length in pixels for that resolution. All configured paths are relative to the repository root.

The example config's `camera_path` supplies camera-relative yaw and translation steps. Positive yaw turns right; translation is `(right, down, forward)` in the current camera frame, scaled by `camera_step_relative` times the first keyframe's median depth. A caller can also use `WorldGenerator.init_scene(name)`, `generate_step(camera)`, and `render(camera)` with arbitrary PyTorch3D camera poses after setting up the upstream import paths as in `run.py`. Future camera poses are not required by OAP.

## Core implementation

| File | Role |
| --- | --- |
| `amodalexplorer/occlusion.py` | Find semantic components with farther surrounding depth; exclude ground and sky. |
| `amodalexplorer/peeling.py` | Complete every candidate per level, apply GAL, enforce depth order, and stop at the layer or mask-area limit. |
| `amodalexplorer/lifting.py` | Fit slanted segment planes using guided depth points, with median-depth anchoring and a frontal-plane fallback. |
| `amodalexplorer/pipeline.py` | Initialize and expand the world; retain all recursive layers and optimize new Gaussians far-to-near. |
| `patches/wonderworld.patch` | Connect upstream depth refinement to GAL, handle an empty sky mask, and redirect sky caching. |

Defaults are three hidden layers, a 0.5% stopping mask area, and a `1e-4` depth push-back when a prediction is in front of its occluder. The stopping test runs after the completed layer is retained, matching the research implementation. GAL fits the 20–80% disparity range, preserves the segment's median-depth anchor, and falls back when more than 10% of projected depths leave half to twice the anchor depth. Gaussian positions remain fixed during optimization, following WonderWorld.

## Validation status

The local release has passed CPU checks comparing candidate extraction and GAL numerically with the research implementation, mocked-prior OAP checks, recursive layer persistence checks, camera-step checks, and clean-checkout patch verification. These checks are development validation, not paper evaluation.

End-to-end CUDA generation and a fresh network installation have **not** been verified for this release: the preparation machine has no working NVIDIA driver. `python run.py --check` reports local dependency and GPU availability without downloading model weights.

## Attribution and license

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for code provenance and inherited terms. [LICENSE](LICENSE) applies to original AmodalExplorer contributions; it does not relicense WonderWorld, its dependencies, or model weights. The upstream checkout is downloaded separately and excluded from this repository and release archive.

Please cite WonderWorld when using the underlying generator:

```bibtex
@article{yu2024wonderworld,
  title={WonderWorld: Interactive 3D Scene Generation from a Single Image},
  author={Hong-Xing Yu and Haoyi Duan and Charles Herrmann and William T. Freeman and Jiajun Wu},
  journal={arXiv:2406.09394},
  year={2024}
}
```

AmodalExplorer paper: *AmodalExplorer: Training-Free Online 3D World Generation via Occlusion-Ordered Completion*, Tianyu Chen, Kang Han, Lu Yu, Di Wu, Yi-Ping Phoebe Chen, Gaowen Liu, Ramana Rao Kompella, and Wei Xiang. Publication metadata will be added when available.
