"""Web extension guards: syntax, import depth, and that every node id / widget name it names exists."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pack import load_pack  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
M = load_pack().NODE_CLASS_MAPPINGS


@pytest.mark.parametrize("path", sorted(WEB.glob("*.js")), ids=lambda p: p.name)
def test_js_parses(path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    out = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_core_imports_resolve_from_the_extension_folder():
    # web/ is served at /extensions/<pack>/, so ComfyUI's scripts are two levels up
    for path in WEB.glob("*.js"):
        for spec in re.findall(r'from\s+"([^"]+)"', path.read_text(encoding="utf-8")):
            if "scripts/" in spec:
                assert spec.startswith("../../scripts/"), (path.name, spec)
            elif spec.startswith("./"):
                assert (WEB / spec[2:]).exists(), (path.name, spec)


def test_widget_rules_name_real_nodes_and_widgets():
    src = (WEB / "ltxe_widgets.js").read_text(encoding="utf-8")
    ids = re.findall(r'"(LTXE_\w+)"', src)
    controls = re.findall(r'control: "([a-z_]+)"', src)
    targets = [n for group in re.findall(r'widgets: \[([^\]]*)\]', src) for n in re.findall(r'"([a-z_]+)"', group)]
    assert ids and controls == ["target", "pad_mode"] and targets == ["custom_width", "custom_height", "pad_value"]
    for nid in ids:
        assert nid in M
        req = M[nid].INPUT_TYPES()["required"]
        for n in controls + targets:
            assert n in req, f"{nid} has no widget {n}"


def test_web_directory_is_registered():
    assert load_pack().WEB_DIRECTORY == "web"
