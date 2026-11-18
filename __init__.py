"""ComfyUI-LTX-Enhanced: resolution, memory and tile planning for LTX video in ComfyUI (Apache-2.0).

Kill switch: environment LTXE_DISABLE=1 registers no nodes (saved workflows then show missing nodes instead of
running something different). LTXE_DISABLE_TILING=1 keeps the nodes but makes LTXE Tiled Model a pass-through.
LTXE_DISABLE_HDR=1 drops only the HDR helpers. The same three switches are settings (Settings > C2C > LTX-Enhanced;
the pack and HDR switches apply at the next start); the variables win when set.
"""
import logging
import os

_LOG = logging.getLogger("ltx_enhanced")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
WEB_DIRECTORY = "web"


def _enabled(setting_id: str, env: str) -> bool:
    """Load-time switch (L3.02): the environment variable (set and not "0" = off) wins, else the C2C setting
    (Settings > C2C > LTX-Enhanced, read at start-up: a change needs a restart), else on."""
    raw = os.environ.get(env, "")
    if raw != "":
        return raw.strip() in ("", "0")
    try:
        from ._c2c_saved_setting import get as _c2c_setting
    except Exception:  # noqa: BLE001 - a settings reader must never cost the nodes
        return True
    return bool(_c2c_setting(setting_id, True))


if not __package__:
    # imported as a plain module (a test runner collecting the folder), not by ComfyUI: nothing to register
    _LOG.debug("ComfyUI-LTX-Enhanced: imported outside a package; nodes not loaded")
elif _enabled("c2c.ltxe.enabled", "LTXE_DISABLE"):
    # Guarded: an import failure here must cost this pack's nodes only, never break ComfyUI's start-up.
    try:
        from .experimental.nodes import NODE_CLASS_MAPPINGS as _EXP, NODE_DISPLAY_NAME_MAPPINGS as _EXP_NAMES

        NODE_CLASS_MAPPINGS.update(_EXP)
        NODE_DISPLAY_NAME_MAPPINGS.update(_EXP_NAMES)
    except Exception as exc:  # noqa: BLE001
        _LOG.error("ComfyUI-LTX-Enhanced: nodes not loaded: %s", exc)
    # A23 SDR -> HDR helpers: their own kill switch, and their own guard (a failure costs only these nodes)
    if _enabled("c2c.ltxe.hdr.enabled", "LTXE_DISABLE_HDR"):
        try:
            from .experimental.hdr_nodes import (NODE_CLASS_MAPPINGS as _HDR,
                                                 NODE_DISPLAY_NAME_MAPPINGS as _HDR_NAMES)

            NODE_CLASS_MAPPINGS.update(_HDR)
            NODE_DISPLAY_NAME_MAPPINGS.update(_HDR_NAMES)
        except Exception as exc:  # noqa: BLE001
            _LOG.error("ComfyUI-LTX-Enhanced: HDR nodes not loaded: %s", exc)
else:
    _LOG.info("ComfyUI-LTX-Enhanced: switched off (LTXE_DISABLE or Settings > C2C > LTX-Enhanced); no nodes registered")

# One menu root for every Code2Collapse pack: "🐺 C2C/<pack>/<family>" (see _c2c_menu.py). Applied after the
# mappings are final; guarded: a menu placement must never cost the nodes.
if __package__:
    try:
        from ._c2c_menu import rebrand_v1 as _c2c_menu_rebrand

        _c2c_menu_rebrand(NODE_CLASS_MAPPINGS, "\U0001F3A5 LTX Enhanced", strip=("LTX Enhanced",))
    except Exception as _exc:  # noqa: BLE001
        _LOG.warning("C2C menu root not applied: %s", _exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
