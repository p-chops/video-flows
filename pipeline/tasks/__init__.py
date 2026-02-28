"""Prefect tasks for the video pipeline."""

from .cut import detect_cuts, extract_segment_task, random_segments, segment_at_cuts
from .sequence import (
    concat_clips, shuffle_clips, interleave_clips,
    generate_static, generate_solid, repeat_clip,
)
from .shader import apply_shader, apply_shader_stack, apply_random_shader_stack
from .composite import (
    blend_layers, masked_composite, multi_layer_composite,
    picture_in_picture, chromakey_composite,
)
from .mask import (
    luma_mask, edge_mask, motion_mask, chroma_mask, gradient_mask,
)
from .color import normalize_levels
from .glitch import bitrate_crush
from .time import (
    time_scrub, drift_loop, ping_pong, echo_trail, time_patch,
    slit_scan, temporal_tile, smear, bloom, frame_stack, slip, flow_warp,
)
from .transition import (
    crossfade, luma_wipe, whip_pan, static_burst, flash, transition_sequence,
)
from .transform import mirror, zoom, invert, hue_shift, saturate
