"""Geometry-aware lifting (GAL)."""
import numpy as np


def refine_disparity(disparity, segments, focal, keep_threshold=10, no_refine_mask=None,
                                    existing_mask=None, existing_disp=None, max_ratio=2.0, max_bad=0.1):
    """Fit each eligible segment to a slanted plane using its guided depth points.

    Use the 20--80% disparity range for fitting and preserve the upstream median
    anchor. Fall back to a frontal plane if orientation or depth bounds fail.
    Existing pixels anchor segments spanning previously generated content.
    """
    H, W = disparity.shape
    ys, xs = np.meshgrid(np.arange(H) + 0.5, np.arange(W) + 0.5, indexing="ij")
    rays = np.stack([(xs - W / 2) / focal, (ys - H / 2) / focal, np.ones_like(xs)])
    refined_disparity = disparity.copy()
    for segment in segments:
        mask = segment['segmentation']
        if no_refine_mask is not None and np.logical_and(mask, no_refine_mask).sum() > 20:
            continue
        disp_pixels_old = disparity[mask]
        p80, p20 = np.percentile(disp_pixels_old, 80), np.percentile(disp_pixels_old, 20)
        valid_disp_px = disp_pixels_old[(disp_pixels_old >= p20) & (disp_pixels_old <= p80)]
        if valid_disp_px.std() > keep_threshold:
            continue
        median_val = np.percentile(valid_disp_px, 50)
        if existing_mask is not None:
            if np.logical_and(mask, ~existing_mask).sum() < 5:
                continue
            intersection_mask = np.logical_and(mask, existing_mask)
            if intersection_mask.sum() > 20:
                median_val = np.percentile(existing_disp[intersection_mask], 50)
        refined_disparity[mask] = median_val  # upstream fallback
        vs_all, us_all = np.nonzero(mask)
        keep = (disp_pixels_old >= p20) & (disp_pixels_old <= p80)
        if keep.sum() < 10:
            continue
        pts = (rays[:, vs_all[keep], us_all[keep]] / disp_pixels_old[keep]).T
        _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
        n = vt[-1]
        if n[2] > 0:
            n = -n
        n = n / max(np.linalg.norm(n), 1e-8)
        if n[2] >= -1e-3:  # not facing the camera
            continue
        vs, us = np.nonzero(mask)
        anchor = (1.0 / median_val) * rays[:, int(np.median(vs)), int(np.median(us))]
        denom = np.einsum("c,cn->n", n, rays[:, mask])
        d = np.where(denom < -1e-6, float(n @ anchor) / np.where(denom < -1e-6, denom, -1.0), np.inf)
        d0 = 1.0 / median_val
        ok = (d > d0 / max_ratio) & (d < d0 * max_ratio)
        if (~ok).mean() > max_bad:
            continue
        refined_disparity[mask] = np.where(ok, 1.0 / np.where(ok, d, d0), median_val)
    return refined_disparity
