"""Prefect flows composing pipeline tasks."""


def __getattr__(name):
    """Lazy imports to avoid triggering module execution on -m runs."""
    _exports = {
        "show_reel": (".show_reel", "show_reel"),
        "show_reel_render": (".show_reel", "show_reel_render"),
        "show_reel_batch": (".show_reel", "show_reel_batch"),
        "batch_shows": (".show_reel", "batch_shows"),
        "brain_wipe": (".brain_wipe", "brain_wipe"),
        "warp_chain": (".brain_wipe", "warp_chain"),
        "brain_wipe_render": (".brain_wipe", "brain_wipe_render"),
        "stooges_channels": (".stooges", "stooges_channels"),
        "evolve_stacks": (".evolve_stacks", "evolve_stacks"),
    }
    if name in _exports:
        mod_name, attr = _exports[name]
        from importlib import import_module
        mod = import_module(mod_name, __package__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
