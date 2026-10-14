"""ComfyUI-LTX-Enhanced: resolution, memory and tile planning for LTX video in ComfyUI (Apache-2.0).

Kill switch: environment LTXE_DISABLE=1 registers no nodes (saved workflows then show missing nodes instead of
running something different). LTXE_DISABLE_TILING=1 keeps the nodes but makes LTXE Tiled Model a pass-through.
"""
import logging
import os

_LOG = logging.getLogger("ltx_enhanced")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

if os.environ.get("LTXE_DISABLE", "").strip() in ("", "0"):
    # Guarded: an import failure here must cost this pack's nodes only, never break ComfyUI's start-up.
    try:
        from .experimental.nodes import NODE_CLASS_MAPPINGS as _EXP, NODE_DISPLAY_NAME_MAPPINGS as _EXP_NAMES

        NODE_CLASS_MAPPINGS.update(_EXP)
        NODE_DISPLAY_NAME_MAPPINGS.update(_EXP_NAMES)
    except Exception as exc:  # noqa: BLE001
        _LOG.error("ComfyUI-LTX-Enhanced: nodes not loaded: %s", exc)
else:
    _LOG.info("ComfyUI-LTX-Enhanced: LTXE_DISABLE is set; no nodes registered")

# One menu root for every Code2Collapse pack: "🐺 C2C/<pack>/<family>" (see _c2c_menu.py). Applied after the
# mappings are final; guarded: a menu placement must never cost the nodes.
try:
    from ._c2c_menu import rebrand_v1 as _c2c_menu_rebrand

    _c2c_menu_rebrand(NODE_CLASS_MAPPINGS, "\U0001F3A5 LTX Enhanced", strip=("LTX Enhanced",))
except Exception as _exc:  # noqa: BLE001
    _LOG.warning("C2C menu root not applied: %s", _exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
