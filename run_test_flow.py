"""
Self-contained test flow: generates two short test videos and blends them.
No external source footage or shaders needed.
"""
from pathlib import Path
from prefect import flow
from pipeline.config import Config
from pipeline.tasks import generate_static, generate_solid, blend_layers


@flow(name="test-blend-demo", log_prints=True)
def test_blend_demo() -> Path:
    cfg = Config()
    cfg.ensure_dirs()

    print("Generating static noise video...")
    static = generate_static(
        cfg.work_dir / "test_static.mp4",
        duration=3.0, width=640, height=480, fps=30.0, cfg=cfg,
    )

    print("Generating solid colour video...")
    solid = generate_solid(
        cfg.work_dir / "test_solid.mp4",
        duration=3.0, color=(20, 80, 180),
        width=640, height=480, fps=30.0, cfg=cfg,
    )

    print("Blending layers (screen mode)...")
    out = cfg.output_dir / "test_blend.mp4"
    blend_layers(static, solid, out, mode="screen", opacity=0.6, cfg=cfg)

    print(f"Done! Output: {out}")
    return out


if __name__ == "__main__":
    result = test_blend_demo()
    print(f"Output at: {result}")
