"""Repository guards: the pack loads cleanly, the research record is complete, and the clean-room boundary holds."""
import hashlib
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
LABELS = ("experimental", "benchmarked", "validated", "production-ready")

# Python modules of Lightricks/ComfyUI-LTXVideo and Lightricks/LTX-2 (LTX Community Licenses). The pack meets them
# in the node graph only (docs/licensing/register.md, D-1): importing or looking them up in sys.modules is forbidden.
RESTRICTED_MODULES = ("ltx_core", "ltx_pipelines", "ltx_kernels", "ltx_trainer", "ltx_video", "comfyui_ltxvideo",
                      "ComfyUI-LTXVideo", "ComfyUI_LTXVideo", "LTXVideo")


def _load_pack():
    spec = importlib.util.spec_from_file_location("ltx_enhanced_pack_under_test", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _own_python_files():
    skip = {"tests"}
    return [p for p in ROOT.rglob("*.py") if not (set(p.relative_to(ROOT).parts) & skip)
            and "__pycache__" not in p.parts]


def test_pack_imports_without_comfyui():
    mod = _load_pack()  # this test process has no ComfyUI on sys.path
    assert isinstance(mod.NODE_CLASS_MAPPINGS, dict)
    assert isinstance(mod.NODE_DISPLAY_NAME_MAPPINGS, dict)
    assert set(mod.NODE_DISPLAY_NAME_MAPPINGS) <= set(mod.NODE_CLASS_MAPPINGS)


def test_every_node_carries_a_stability_label():
    mod = _load_pack()
    for node_id, cls in mod.NODE_CLASS_MAPPINGS.items():
        desc = str(getattr(cls, "DESCRIPTION", "") or "")
        assert any(lbl in desc.lower() for lbl in LABELS), f"{node_id}: DESCRIPTION has no stability label"


def test_licence_is_the_apache_2_text():
    assert hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == APACHE_SHA256


def test_registry_is_complete_and_rendered():
    spec = importlib.util.spec_from_file_location("ltx_registry_tool", ROOT / "tools" / "registry.py")
    reg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reg)
    assert reg.problems() == []


def test_no_restricted_upstream_imports():
    names = "|".join(re.escape(n) for n in RESTRICTED_MODULES)
    pat = re.compile(rf"^\s*(from|import)\s+[.\w-]*({names})\b|sys\.modules\s*\[\s*['\"][^'\"]*({names})",
                     re.M | re.I)
    hits = []
    for p in _own_python_files():
        for m in pat.finditer(p.read_text(encoding="utf-8")):
            hits.append(f"{p.relative_to(ROOT).as_posix()}: {m.group(0).strip()}")
    assert hits == [], "graph-level integration only (register D-1): " + "; ".join(hits)


def test_restricted_pattern_catches_the_forms_it_must():
    names = "|".join(re.escape(n) for n in RESTRICTED_MODULES)
    pat = re.compile(rf"^\s*(from|import)\s+[.\w-]*({names})\b|sys\.modules\s*\[\s*['\"][^'\"]*({names})",
                     re.M | re.I)
    for bad in ("from ltx_core.model import X", "import ltx_pipelines", "x = sys.modules['custom_nodes.ComfyUI-LTXVideo']",
                "    from custom_nodes.comfyui_ltxvideo import tiled_sampler"):
        assert pat.search(bad), bad
    for ok in ("import torch", "from comfy.ldm.lightricks import model", "# see Lightricks/ComfyUI-LTXVideo for behaviour"):
        assert not pat.search(ok), ok
