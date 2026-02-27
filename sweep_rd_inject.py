"""
Sweep inject_density for the RD shader.

Renders 30s at each parameter value with all other params at defaults.
Finer spacing at the low end where gradual pattern formation happens.

Output: output/rd_inject_sweep/rd_inject_XXXX.mp4
"""
from pathlib import Path

from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner

from pipeline.config import Config
from pipeline.tasks import generate_solid, apply_shader_stack

RD_SHADER = Path("brain-wipe-shaders/ulp-brain-wipe-rd.fs")

RD_DEFAULTS = {
    "feed": 0.037,
    "kill": 0.060,
    "diff_a": 0.82,
    "diff_b": 0.41,
    "color_speed": 0.12,
    "color_shift": 0.0,
    "sim_scale": 0.003,
    "burst": 0.0,
    "reset": 0.0,
}

SWEEP = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5, 0.75, 1.0]


@flow(name="rd-inject-sweep", log_prints=True,
      task_runner=ConcurrentTaskRunner(max_workers=4))
def rd_inject_sweep(
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    duration: float = 30.0,
):
    cfg = Config()
    cfg.ensure_dirs()

    out_dir = Path("output/rd_inject_sweep_rescaled")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate one shared placeholder clip
    solid = cfg.work_dir / "rd_sweep_solid.mp4"
    generate_solid(solid, duration, width=width, height=height, fps=fps, cfg=cfg)

    # Submit all sweep values in parallel
    futures = []
    for val in SWEEP:
        params = dict(RD_DEFAULTS)
        params["inject_density"] = val

        tag = f"{val:.6f}".replace(".", "p")
        dst = out_dir / f"rd_inject_{tag}.mp4"

        print(f"Submitting inject_density={val:.4f} -> {dst.name}")
        f = apply_shader_stack.submit(
            solid, dst, [RD_SHADER],
            param_overrides={"ulp-brain-wipe-rd": params},
            cfg=cfg,
        )
        futures.append((val, dst, f))

    # Wait for all renders
    for val, dst, f in futures:
        f.result()
        print(f"  done: inject_density={val:.4f} -> {dst.name}")

    print(f"\n{len(futures)} renders complete -> {out_dir}/")


if __name__ == "__main__":
    rd_inject_sweep()
