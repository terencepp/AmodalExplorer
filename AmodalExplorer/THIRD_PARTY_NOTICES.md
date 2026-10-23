# Third-party provenance

This repository builds on [WonderWorld](https://github.com/KovenYu/WonderWorld) at commit `5cf11467ad21a9d7bca269f961855a235c7d352e`. Upstream source, assets, native extensions, and pretrained weights are obtained separately and are not included in the release archive.

- `amodalexplorer/pipeline.py`: the generation flow and Gaussian optimization adapt WonderWorld's `run.py`; the AmodalExplorer controller retains recursive OAP layers and connects GAL to lifting.
- `amodalexplorer/prompts.py`: local text formatting adapts `TextpromptGen.generate_keywords` and `generate_prompt` from WonderWorld's `util/chatGPT4.py`. Network calls and API clients are excluded.
- `amodalexplorer/lifting.py`: AmodalExplorer's depth-oriented segment-plane fitting extends the segment selection and median anchoring in WonderWorld's `util/segment_utils.py`.
- `patches/wonderworld.patch`: minimal changes to WonderWorld's `models/models.py` for GAL and cache integration.
- `config/examples.yaml`: a caption and image reference from WonderWorld's example metadata. The referenced image stays in the separately obtained upstream checkout.

Keep the original notices in the prepared upstream checkout. In particular, the pinned upstream includes files with Seoul National University copyright and permission notices; the Gaussian rasterizer includes a Gaussian-Splatting license with noncommercial conditions. The pinned checkout does not contain a repository-wide `LICENSE` granting blanket rights to all its code. This release does not replace those notices or grant additional rights to upstream material. Its MIT license covers only original AmodalExplorer contributions to the extent their authors hold the relevant rights.

Frozen priors and their weights retain their respective terms: [Stable Diffusion](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting), [Marigold](https://github.com/prs-eth/Marigold), [OneFormer](https://github.com/SHI-Labs/OneFormer), [RepViT](https://github.com/THU-MIG/RepViT), and [SyncDiffusion](https://github.com/KAIST-Visual-AI-Group/SyncDiffusion).
