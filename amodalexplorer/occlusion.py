"""Candidate occluders and the background-extension depth used before a layer is generated.

A candidate is a connected component of one semantic class (OneFormer ADE20K, as in upstream
`generate_layer`) that is in front of what surrounds it: a large enough part of its outer ring
is farther away. Ground classes and sky are never candidates. Candidates are re-extracted after each completion.
"""
import numpy as np
import torch
from scipy.ndimage import label, binary_dilation, binary_erosion

# ADE20K ids treated as ground in upstream generate_layer (floor, road, grass, pavement, earth,
# sea, field, sand, path, lake).
GROUND_IDS = {3, 6, 9, 11, 13, 26, 29, 46, 52, 128}
SKY_ID = 2


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def find_candidates(sem_map, disparity, sky_mask, region_mask=None, cfg=None):
    """
    sem_map:     [H, W] int semantic labels.
    disparity:   [H, W] disparity of the current (front-most) surface.
    sky_mask:    [H, W] bool.
    region_mask: [H, W] bool or None. OAP only peels occluders inside the newly generated
                 region (upstream writes geometry only inside `outpaint_mask`).
    Returns a list of dicts with keys: mask [H, W] bool, ring_far [H, W] bool, gap (relative
    disparity gap), area (fraction of the image), disp_in.
    """
    cfg = cfg or {}
    min_area = float(cfg.get("min_area", 0.001))
    ring_px = int(cfg.get("ring_px", 6))
    rel_gap = float(cfg.get("rel_gap", 0.1))
    min_far_frac = float(cfg.get("min_far_frac", 0.2))
    min_region_overlap = float(cfg.get("min_region_overlap", 0.5))

    sem = sem_map.cpu().numpy() if torch.is_tensor(sem_map) else np.asarray(sem_map)
    disp = disparity.cpu().numpy() if torch.is_tensor(disparity) else np.asarray(disparity)
    sky = sky_mask.cpu().numpy().astype(bool) if torch.is_tensor(sky_mask) else np.asarray(sky_mask, bool)
    region = None
    if region_mask is not None:
        region = region_mask.cpu().numpy().astype(bool) if torch.is_tensor(region_mask) else np.asarray(region_mask, bool)
    hw = sem.shape[0] * sem.shape[1]
    ring_k, band_k = _disk(ring_px), _disk(max(ring_px // 2, 1))

    cands = []
    for cls in np.unique(sem):
        if int(cls) in GROUND_IDS or int(cls) == SKY_ID:
            continue
        comps, n = label(sem == cls)
        for i in range(1, n + 1):
            comp = comps == i
            area = comp.sum() / hw
            if area < min_area:
                continue
            if region is not None and (comp & region).sum() < min_region_overlap * comp.sum():
                continue
            ring = binary_dilation(comp, ring_k) & ~comp & ~sky
            band = comp & ~binary_erosion(comp, band_k)
            if ring.sum() == 0 or band.sum() == 0:
                continue
            disp_in = float(np.median(disp[band]))
            far = ring & (disp < disp_in * (1 - rel_gap))
            if far.sum() < min_far_frac * ring.sum():
                continue
            gap = (disp_in - float(np.median(disp[far]))) / disp_in
            cands.append({"mask": comp, "ring_far": far, "gap": gap, "area": area, "disp_in": disp_in, "cls": int(cls)})
    return cands



def inpaint_mask_from(cands, shape, grow_px=2):
    """Union of candidate masks, grown to cover object borders."""
    mask = np.zeros(shape, dtype=bool)
    for candidate in cands:
        mask |= candidate["mask"]
    if grow_px > 0:
        mask = binary_dilation(mask, _disk(grow_px))
    return mask
