"""Occlusion-ordered amodal peeling (OAP)."""
import numpy as np
import torch
from kornia.morphology import dilation

from util.utils import soft_stitching
from amodalexplorer.occlusion import find_candidates, inpaint_mask_from

MAX_VALID_DEPTH = 0.006 * 0.8


class AmodalPeeler:
    def __init__(self, config, keyframe_generator):
        self.kf = keyframe_generator
        self.max_layers = int(config.get("max_layers", 3))
        self.stop_ratio = float(config.get("stop_ratio", 0.005))
        self.grow_px = int(config.get("grow_px", 2))
        self.cand_cfg = dict(config.get("candidates", {}))
        self.negative_prompt = config.get("negative_prompt", "tree, plant")

    @torch.no_grad()
    def run(self, outpaint_mask, sem_seg, ground_mask, scene_name):
        """Complete hidden layers and return images, depths and masks for persistent lifting."""
        kf = self.kf
        dev = kf.depth_latest.device
        images = [kf.image_latest.clone()]
        depths = [kf.depth_latest.clone()]
        masks = []
        h, w = depths[0].shape[-2:]
        region = outpaint_mask[0, 0].bool()
        ground_np = ground_mask.squeeze().cpu().numpy()
        sem, sky = sem_seg, kf.sky_mask_latest[0, 0].bool()

        for level in range(self.max_layers):
            if level > 0:
                kf.image_latest = images[-1]
                sem = kf.update_sky_mask()
                sky = kf.sky_mask_latest[0, 0].bool()
            disp = 1.0 / depths[-1][0, 0]
            cands = find_candidates(sem, disp, sky, region_mask=region, cfg=self.cand_cfg)
            if not cands:
                break
            occ = torch.from_numpy(inpaint_mask_from(cands, (h, w), grow_px=self.grow_px)).to(dev)[None, None]
            prev_img, prev_depth = images[-1], depths[-1]

            # 1 completion call: inpaint the whole occluder mask, then guided depth (as upstream).
            kf.inpaint(prev_img, inpaint_mask=occ, inpainting_prompt=scene_name, negative_prompt=self.negative_prompt,
                       mask_strategy=np.max, diffusion_steps=50)
            stitch = dilation(occ.float(), kernel=torch.ones(5, 5, device=dev))
            new_img = soft_stitching(kf.image_latest, prev_img, stitch, sigma=1, blur_size=3)
            kf.image_latest = new_img
            in_range = prev_depth < MAX_VALID_DEPTH
            kf.get_depth(new_img, archive_output=True, target_depth=prev_depth, mask_align=~occ & in_range,
                         mask_farther=occ & in_range, diffusion_steps=30, guidance_steps=8)
            kf.refine_disp_with_segments(no_refine_mask=ground_np, existing_mask=~occ.squeeze().cpu().numpy(),
                                         existing_disp=(1.0 / prev_depth).squeeze().cpu().numpy())
            new_depth = kf.depth_latest.clone()
            wrong = new_depth < prev_depth
            new_depth[wrong] = prev_depth[wrong] + 0.0001  # hidden content must be farther
            new_depth = torch.where(occ, new_depth, prev_depth)

            images.append(new_img)
            depths.append(new_depth)
            masks.append(occ)
            if float(occ.float().mean()) < self.stop_ratio:
                break

        return {"images": images, "depths": depths, "masks": masks, "calls": len(masks)}
