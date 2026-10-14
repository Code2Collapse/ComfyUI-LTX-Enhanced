"""Put ComfyUI core on sys.path when it is available, for the tests that check integration with it.

Order: env LTXE_COMFY_CORE, then the owner / CI layout (Custom_Nodes/<pack> beside ComfyUI_windows_portable/ComfyUI).
Tests that need core use pytest.importorskip, so the planning tests run without it.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CANDIDATES = [Path(p) for p in (os.environ.get("LTXE_COMFY_CORE", ""),) if p] + [
    ROOT.parent.parent / "ComfyUI_windows_portable" / "ComfyUI"]
for _c in _CANDIDATES:
    if (_c / "comfy_api").is_dir():
        if str(_c) not in sys.path:
            sys.path.insert(0, str(_c))
        break
