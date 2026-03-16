#!/usr/bin/env python
"""Demo: SMPL body model — pure NumPy forward pass examples.

Shows the SMPL model loading, shape variation (prompt → betas),
and posed mesh generation. No smplx/torch needed.

Run:
    python scripts/demo_smpl.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modules.physics.smpl_body import SMPLBody, is_smpl_available, smpl_betas_from_body_params
from src.modules.physics.body_model import BodyParams, body_params_from_prompt


def main() -> None:
    print("=" * 60)
    print("  SMPL Body Model — Working Examples")
    print("=" * 60)

    # 1. Availability check
    print("\n[1] SMPL Availability Check")
    print(f"    is_smpl_available() = {is_smpl_available()}")

    if not is_smpl_available():
        print("    SMPL model files not found — run scripts/setup_smpl.py")
        return

    # 2. Load model
    print("\n[2] Load SMPL Model")
    body = SMPLBody.get_or_create(gender="neutral")
    print(f"    Vertices:  {body.n_vertices}")
    print(f"    Faces:     {body.n_faces}")
    h0 = body.rest_vertices[:, 1].max() - body.rest_vertices[:, 1].min()
    print(f"    Template height: {h0:.3f} m")

    # 3. Rest-pose mesh
    print("\n[3] Rest-Pose (T-Pose) Mesh")
    verts, faces = body.get_shaped_mesh()
    print(f"    Vertex range X: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}] m")
    print(f"    Vertex range Y: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}] m")
    print(f"    Vertex range Z: [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}] m")

    # 4. Shape variation
    print("\n[4] Shape Variation (beta0 controls overall size)")
    for beta_val in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        betas = np.zeros(10, dtype=np.float32)
        betas[0] = beta_val
        v, _ = body.get_shaped_mesh(betas=betas)
        h = v[:, 1].max() - v[:, 1].min()
        print(f"    beta0={beta_val:+5.1f}  →  height={h:.3f} m")

    # 5. Prompt → BodyParams → SMPL betas
    print("\n[5] Prompt → BodyParams → SMPL Betas → Mesh Height")
    prompts = [
        "a tall muscular man",
        "a short thin woman",
        "a child running",
        "an average person",
        "an elderly woman",
    ]
    for p in prompts:
        params = body_params_from_prompt(p)
        betas = smpl_betas_from_body_params(params)
        v, _ = body.get_shaped_mesh(betas=betas)
        h = v[:, 1].max() - v[:, 1].min()
        print(f'    "{p}"')
        print(
            f"      height={params.height_m:.2f}m  "
            f"betas[:3]=[{betas[0]:.2f}, {betas[1]:.2f}, {betas[2]:.2f}]  "
            f"mesh_h={h:.3f}m"
        )

    # 6. Forward pass with pose (arms down)
    print("\n[6] Forward Pass with Pose (Arms Down)")
    body_pose = np.zeros(69, dtype=np.float64)
    # Rotate left shoulder (joint 16) around Z axis
    body_pose[(16 - 1) * 3 + 2] = 1.2   # ~70° down
    # Rotate right shoulder (joint 17) around Z axis
    body_pose[(17 - 1) * 3 + 2] = -1.2  # ~70° down
    verts_posed, joints = body.forward(body_pose=body_pose)
    print(f"    Posed vertices shape: {verts_posed.shape}")
    print(f"    Posed joints shape:   {joints.shape}")
    print(f"    Pelvis:  [{joints[0, 0]:.4f}, {joints[0, 1]:.4f}, {joints[0, 2]:.4f}]")
    print(f"    Head:    [{joints[15, 0]:.4f}, {joints[15, 1]:.4f}, {joints[15, 2]:.4f}]")
    print(f"    L_Hand:  [{joints[20, 0]:.4f}, {joints[20, 1]:.4f}, {joints[20, 2]:.4f}]")
    print(f"    R_Hand:  [{joints[21, 0]:.4f}, {joints[21, 1]:.4f}, {joints[21, 2]:.4f}]")

    # 7. Silhouette rendering demo
    print("\n[7] Silhouette Rendering (SMPL mesh → 2D image)")
    try:
        import cv2
        from src.modules.physics.smpl_body import render_smpl_silhouette

        canvas = np.full((512, 512, 3), 230, dtype=np.uint8)

        # Simple orthographic projection
        def project(xyz: np.ndarray) -> np.ndarray:
            """mm -> screen pixels via orthographic projection."""
            scale = 0.25
            cx, cy = 256, 280
            col = xyz[:, 0] * scale + cx
            row = -xyz[:, 1] * scale + cy
            depth = xyz[:, 2]
            return np.stack([col, row, depth], axis=-1)

        # Get a shaped mesh (tall muscular man) in mm
        params = body_params_from_prompt("a tall muscular man")
        betas = smpl_betas_from_body_params(params)
        verts_m, faces = body.get_shaped_mesh(betas=betas)
        verts_mm = verts_m * 1000.0  # metres → mm

        result = render_smpl_silhouette(verts_mm, faces, project, canvas)

        out_path = str(Path(__file__).resolve().parents[1] / "outputs" / "smpl_silhouette_demo.png")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, result)
        print(f"    Saved silhouette: {out_path}")
        print(f"    Image shape: {result.shape}")
    except Exception as e:
        print(f"    Silhouette render skipped: {e}")

    print("\n" + "=" * 60)
    print("  All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
