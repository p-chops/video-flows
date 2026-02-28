"""Prefect flows composing pipeline tasks."""

from .examples import (
    cut_shuffle_shader,
    random_shader_collage,
    density_composite,
    masked_shader_overlay,
    texture_builder,
    shuffled_scene_shaders,
    deep_color,
)
from .stooges import stooges_channels
from .brain_wipe import (
    brain_wipe, warp_chain, brain_wipe_render, filter_shaders,
    pick_shader_stack, LEVEL_PARAMS,
)
from .compositing_lab import compositing_lab
from .time_lab import time_lab, time_lab_scrub, time_lab_drift, time_lab_pingpong, time_lab_echo, time_lab_patch
from .transition_lab import (
    transition_lab, transition_lab_crossfade, transition_lab_luma_wipe,
    transition_lab_whip_pan, transition_lab_static_burst,
    transition_lab_flash, transition_lab_sequence,
)
