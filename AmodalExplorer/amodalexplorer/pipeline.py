"""Persistent online world generation using WonderWorld priors, OAP and GAL.

Generation and Gaussian optimization are adapted from WonderWorld run.py.
See THIRD_PARTY_NOTICES.md for provenance and upstream terms.
"""
import gc
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from kornia.morphology import dilation

from arguments import GSParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss import l1_loss, ssim
from models.models import KeyframeGen
from util.utils import load_example_yaml, convert_pt3d_cam_to_3dgs_cam, soft_stitching

from amodalexplorer.peeling import AmodalPeeler
from amodalexplorer.prompts import PromptBuilder

XYZ_SCALE = 1000
MAX_VALID_DEPTH = 0.006 * 0.8
MIN_STAGE_POINTS = 16  # a layer with fewer surviving points is skipped (degenerate, invisible)


def seeding(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                torch.cuda.synchronize()
                self.s = time.time()

            def __exit__(self, *a):
                torch.cuda.synchronize()
                timer.t[name] = timer.t.get(name, 0.0) + time.time() - self.s
        return _Ctx()


def load_models(config):
    """Load frozen image, depth, normal and segmentation priors."""
    from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
    from diffusers import DDIMScheduler, EulerDiscreteScheduler
    from diffusers.models.attention_processor import AttnProcessor2_0
    from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldNormalsPipeline
    from util.segment_utils import create_mask_generator_repvit
    from util.utils import prepare_scheduler

    dev = config["device"]
    m = {}
    m["segment_processor"] = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
    m["segment_model"] = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_large").to(dev)
    m["mask_generator"] = create_mask_generator_repvit()
    inp = StableDiffusionInpaintPipeline.from_pretrained(config["stable_diffusion_checkpoint"], safety_checker=None,
                                                         torch_dtype=torch.bfloat16).to(dev)
    inp.scheduler = DDIMScheduler.from_config(inp.scheduler.config)
    inp.unet.set_attn_processor(AttnProcessor2_0())
    inp.vae.set_attn_processor(AttnProcessor2_0())
    m["inpainter_pipeline"] = inp
    depth = MarigoldPipeline.from_pretrained("prs-eth/marigold-depth-v1-0", torch_dtype=torch.bfloat16).to(dev)
    depth.scheduler = prepare_scheduler(EulerDiscreteScheduler.from_config(depth.scheduler.config))
    m["depth_model"] = depth
    m["normal_estimator"] = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v0-1", torch_dtype=torch.bfloat16).to(dev)
    return m


def train_gaussian(gaussians, scene, opt, background, initialize_scaling=True):
    """Optimize new Gaussians while keeping earlier layers frozen."""
    from random import randint
    train_cams = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=train_cams, initialize_scaling=initialize_scaling)
    for iteration in range(1, opt.iterations + 1):
        stack = scene.getTrainCameras().copy()
        cam = stack.pop(randint(0, len(stack) - 1))
        pkg = render(cam, gaussians, opt, background)
        image, vs_points, vis, radii = pkg["render"], pkg["viewspace_points"], pkg["visibility_filter"], pkg["radii"]
        gt = cam.original_image.cuda()
        photo = (1.0 - opt.lambda_dssim) * l1_loss(image, gt) + opt.lambda_dssim * (1.0 - ssim(image, gt))
        loss = photo
        loss.backward()
        n = gaussians.get_xyz.shape[0]
        grad, vis, radii = vs_points.grad[:n], vis[:n], radii[:n]
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[vis] = torch.max(gaussians.max_radii2D[vis], radii[vis])
                gaussians.add_densification_stats(grad, vis)
                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    max_screen = opt.max_screen_size if iteration >= opt.prune_from_iter else None
                    extent = 0.0003 * XYZ_SCALE * 2 if opt.scene_extent is None else opt.scene_extent
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, extent, max_screen)
                    gaussians.compute_3D_filter(cameras=train_cams)
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)


class WorldGenerator:
    def __init__(self, config, models):
        self.config = config
        self.models = models
        self.config["gen_layer"] = False  # OAP manages hidden layers
        self.dev = config["device"]
        self.background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device="cuda")
        self.opt = GSParams()
        self.kf = None
        self.gaussians = None
        self.peeler = None

    # ------------------------------------------------------------------ helpers
    def render(self, cam):
        with torch.no_grad():
            pkg = render(convert_pt3d_cam_to_3dgs_cam(cam, xyz_scale=XYZ_SCALE), self.gaussians, self.opt, self.background)
        return {"rgb": pkg["render"].clamp(0, 1), "alpha": pkg["final_opacity"][0], "depth": pkg["median_depth"][0] / XYZ_SCALE}

    def _traindata(self, pc, image):
        """One-frame training data for the current camera (cf. convert_to_3dgs_traindata_latest)."""
        kf = self.kf
        w2c = kf.cameras_archive[-1].get_world_to_view_transform().get_matrix()[0].transpose(0, 1).clone()
        w2c[:3, 3] *= XYZ_SCALE
        c2w = w2c.inverse() @ torch.diag(torch.tensor([-1.0, 1, -1, 1], device=self.dev))
        frame = {"image": ToPILImage()(image[0].clamp(0, 1)), "transform_matrix": c2w.cpu().numpy().tolist(), "no_loss_mask": None}
        return {"frames": [frame], "pcd_points": pc["xyz"].permute(1, 0).cpu().numpy() * XYZ_SCALE,
                "pcd_colors": pc["rgb"].cpu().numpy(), "pcd_normals": pc["normals"].cpu().numpy(),
                "camera_angle_x": 2 * np.arctan(512 / (2 * kf.init_focal_length)), "W": 512, "H": 512}

    def _train_all(self, traindatas, timer):
        """Train each layer in order (deepest first); every stage only optimizes its new points."""
        with timer("gs_opt"):
            for td in traindatas:
                if td["pcd_points"].shape[-1] == 0:
                    continue
                g = GaussianModel(sh_degree=0, previous_gaussian=self.gaussians)
                scene = Scene(td, g, self.opt)  # builds the gaussians from the point cloud
                if g.get_xyz.shape[0] < MIN_STAGE_POINTS:
                    # (almost) every point was removed by the floater filter; upstream crashes here
                    # (compute_3D_filter takes max() over an empty set when no point is in front of the camera)
                    continue
                train_gaussian(g, scene, self.opt, self.background)
                self.gaussians = g

    # ------------------------------------------------------------------ scene setup
    def init_scene(self, example_name):
        """Initialize a persistent Gaussian world from an image and its description."""
        cfg = self.config
        seeding(cfg["seed"])
        m = self.models
        kf = KeyframeGen(config=cfg, inpainter_pipeline=m["inpainter_pipeline"], mask_generator=m["mask_generator"],
                         depth_model=m["depth_model"], segment_model=m["segment_model"],
                         segment_processor=m["segment_processor"], normal_estimator=m["normal_estimator"],
                         rotation_path=cfg["rotation_path"][: cfg["num_scenes"]],
                         inpainting_resolution=cfg["inpainting_resolution_gen"]).to(self.dev)
        self.kf = kf
        timer = Timer()

        y = load_example_yaml(example_name, cfg.get("examples_yaml", "examples/examples.yaml"))
        self.style_prompt = y["style_prompt"]
        self.negative = (y["negative_prompt"] + ", ") if y["negative_prompt"] != "" else ""
        content = y["content_prompt"].split(",")
        self.scene_name = content[0]
        self.prompt = PromptBuilder().generate_prompt(y["style_prompt"], content[1:],
                                                       y.get("background"), content[0])
        kf.image_latest = ToTensor()(Image.open(y["image_filepath"]).convert("RGB").resize((512, 512))).unsqueeze(0).to(self.dev)

        sky_dir = Path(cfg["sky_cache_dir"]) / example_name
        # Lock the sky cache so concurrent processes cannot read partially written files.
        import fcntl
        sky_dir.parent.mkdir(parents=True, exist_ok=True)
        sky_lock = open(sky_dir.parent / f"{example_name}.lock", "w")
        fcntl.flock(sky_lock, fcntl.LOCK_EX)
        sync = None
        if cfg["gen_sky_image"] or not ((sky_dir / "sky_0.png").exists() and (sky_dir / "sky_1.png").exists()):
            from syncdiffusion.syncdiffusion_model import SyncDiffusion
            # stabilityai/stable-diffusion-2-inpainting is gone from the Hub: use the configured mirror;
            # hf_key skips the branch that sets `condition`, so set it as the 2.0-inpaint branch would
            sync = SyncDiffusion(self.dev, sd_version="2.0-inpaint", hf_key=cfg["stable_diffusion_checkpoint"])
            sync.condition = True
        sky_mask = kf.generate_sky_mask().float()
        kf.generate_sky_pointcloud(sync, image=kf.image_latest, mask=sky_mask, gen_sky=cfg["gen_sky_image"], style=self.style_prompt)
        del sync

        start_cam = kf.get_camera_at_origin()
        with timer("keyframe"):
            pending = self._init_keyframe(start_cam)
        d = kf.depth_latest
        median_depth = float(d[d < MAX_VALID_DEPTH].median())
        kf.increment_kf_idx()

        # Cache sky Gaussians for subsequent runs of this example.
        sky_ply = sky_dir / "finished_3dgs_sky_tanh.ply"
        if cfg["gen_sky"] or not sky_ply.exists():
            td_sky = kf.convert_to_3dgs_traindata(xyz_scale=XYZ_SCALE, remove_threshold=None, use_no_loss_mask=False)[1]
            g = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
            opt = GSParams()
            opt.max_screen_size, opt.scene_extent = 100, 1.5
            opt.densify_from_iter = opt.prune_from_iter = 200
            opt.densify_grad_threshold, opt.iterations = 1.0, 399
            scene = Scene(td_sky, g, opt, is_sky=True)
            train_gaussian(g, scene, opt, self.background, initialize_scaling=False)
            g.save_ply_with_filter(str(sky_ply))
        else:
            g = GaussianModel(sh_degree=0)
            g.load_ply_with_filter(str(sky_ply))
        fcntl.flock(sky_lock, fcntl.LOCK_UN)  # sky images and sky Gaussians are both cached now
        sky_lock.close()
        n = g.get_xyz_all.shape[0]
        g.visibility_filter_all = torch.zeros(n, dtype=torch.bool, device="cuda")
        g.delete_mask_all = torch.zeros(n, dtype=torch.bool, device="cuda")
        g.is_sky_filter = torch.ones(n, dtype=torch.bool, device="cuda")
        self.gaussians = g

        self._train_all(pending["traindatas"], timer)
        self.gaussians.set_inscreen_points_to_visible(convert_pt3d_cam_to_3dgs_cam(start_cam, xyz_scale=XYZ_SCALE))
        return start_cam, {"median_depth": median_depth, "timing": timer.t, "calls": pending["calls"]}

    def _init_keyframe(self, start_cam):
        """Estimate guided depth before completing and lifting the hidden layers."""
        kf = self.kf
        kf.set_current_camera(start_cam, archive_camera=True)
        sem_map = kf.update_sky_mask()
        render_output = kf.render(render_sky=True)
        kf.image_latest = soft_stitching(render_output["rendered_image"], kf.image_latest, kf.sky_mask_latest)
        ground_mask = kf.generate_ground_mask(sem_map=sem_map)[None, None]
        ground_depth = kf.compute_ground_depth(camera_height=0.0003)
        ground_ok = (ground_depth > 0.001) & (ground_depth < MAX_VALID_DEPTH)
        kf.get_depth(kf.image_latest, archive_output=True, target_depth=ground_depth, mask_align=(ground_mask & ground_ok),
                     diffusion_steps=30, guidance_steps=8)
        kf.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())
        outpaint = torch.ones(1, 1, 512, 512, dtype=torch.bool, device=self.dev)
        return self._write_layers(outpaint, sem_map, ground_mask)

    def _write_layers(self, outpaint_mask, sem_seg, ground_mask):
        """Lift the completed background, then retain each foreground layer far-to-near."""
        kf = self.kf
        if self.peeler is None:
            self.peeler = AmodalPeeler(self.config["oap"], kf)
        # same layer prompt as upstream: "road, building" at the first keyframe, the scene name afterwards
        prompt = self.scene_name if kf.kf_idx > 0 else "road, building"
        res = self.peeler.run(outpaint_mask.bool(), sem_seg, ground_mask, prompt)
        imgs, depths, occs = res["images"], res["depths"], res["masks"]
        n = len(occs)
        region = outpaint_mask.bool()

        # base = deepest completed image; then each peeled occluder from the image in front of it
        kf.image_latest, kf.depth_latest = imgs[n], depths[n]
        kf.update_sky_mask()
        kf.update_current_pc_by_kf(image=imgs[n], depth=depths[n], valid_mask=region & ~kf.sky_mask_latest.bool())
        traindatas = [self._traindata(kf.current_pc_latest, imgs[n])]
        for j in range(n, 0, -1):
            valid = occs[j - 1] & region
            kf.update_current_pc_by_kf(image=imgs[j - 1], depth=depths[j - 1], valid_mask=valid, gen_layer=True)
            traindatas.append(self._traindata(kf.current_pc_layer_latest, imgs[j - 1]))

        kf.image_latest, kf.depth_latest = imgs[0], depths[0]
        kf.archive_latest()
        return {"traindatas": traindatas, "calls": n}

    # ------------------------------------------------------------------ one keyframe
    def generate_step(self, cam):
        """Expand the world at the supplied PyTorch3D camera pose."""
        kf, cfg = self.kf, self.config
        timer = Timer()
        prompt = self.prompt
        kf.set_kf_param(inpainting_resolution=cfg["inpainting_resolution_gen"], inpainting_prompt=prompt,
                        adaptive_negative_prompt=self.negative)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(cam, xyz_scale=XYZ_SCALE)
        kf.set_current_camera(cam, archive_camera=True)

        with timer("outpaint"):
            with torch.no_grad():
                pkg = render(tdgs_cam, self.gaussians, self.opt, self.background)
                pkg_nosky = render(tdgs_cam, self.gaussians, self.opt, self.background, exclude_sky=True)
            side_sky_height, sky_cond_width = 128, 40
            m05_nosky = pkg_nosky["final_opacity"] < 0.6
            m00_nosky = pkg_nosky["final_opacity"] < 0.01
            m05 = pkg["final_opacity"] < 0.6
            m00 = pkg["final_opacity"] < 0.01
            fg_cols = torch.sum(~m05_nosky == 1, dim=1) > 150
            fg_idx = torch.nonzero(fg_cols, as_tuple=True)[1]
            full = torch.zeros(1, 1, 512, 512, device=self.dev)
            if fg_idx.numel() > 0:
                full[:, :, :, fg_idx.min().item():fg_idx.max().item() + 1] = 1
            full[:, :, :sky_cond_width, :] = 1
            full[:, :, :side_sky_height, :sky_cond_width] = 1
            full[:, :, :side_sky_height, -sky_cond_width:] = 1
            nosky = 1 - full
            cond = pkg_nosky["render"] * nosky + pkg["render"] * full
            fill_mask = m05_nosky * nosky + m05 * full
            outpaint_mask = m00_nosky * nosky + m00 * full
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())
            kf.inpaint(cond, inpaint_mask=outpaint_mask, fill_mask=fill_mask, inpainting_prompt=prompt,
                       mask_strategy=np.max, diffusion_steps=50)

            sem_seg = kf.update_sky_mask()
            recomposed = soft_stitching(pkg["render"], kf.image_latest, kf.sky_mask_latest)
            depth_should_be = pkg["median_depth"][0:1].unsqueeze(0) / XYZ_SCALE
            align = (depth_should_be < MAX_VALID_DEPTH) & (depth_should_be > 0.001)
            ground_mask = kf.generate_ground_mask(sem_map=sem_seg)[None, None]
            ground_depth = kf.compute_ground_depth(camera_height=0.0003)
            ground_ok = (ground_depth > 0.001) & (ground_depth < MAX_VALID_DEPTH)
            joint = align | (ground_mask & ground_ok)
            joint_depth = torch.where(align, depth_should_be, ground_depth)
            with torch.no_grad():
                kf.get_depth(kf.image_latest, target_depth=joint_depth, mask_align=joint, archive_output=True,
                             diffusion_steps=30, guidance_steps=8)
            kf.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())
            kf.image_latest = recomposed

        info = {}
        with timer("hidden"):
            out = self._write_layers(outpaint_mask > 0.5, sem_seg, ground_mask)
            info["calls"] = out["calls"]
        self._train_all(out["traindatas"], timer)
        self.gaussians.set_inscreen_points_to_visible(tdgs_cam)
        kf.increment_kf_idx()
        empty_cache()
        info["timing"] = timer.t
        info["outpaint_frac"] = float((outpaint_mask > 0.5).float().mean())
        return info
