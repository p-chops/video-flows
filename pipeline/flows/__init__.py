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
