"""Colour helpers for the HDR pipeline (no ComfyUI import). Written from the published definitions, clean-room.

- ACEScct: the logarithmic encoding of linear AP1 (ACEScg) values, Academy specification S-2016-001. Below the break
  point it is a straight line (so it stays defined at and below 0); above it, (log2(x) + 9.72) / 17.52.
- Rec.709 / sRGB linear -> AP1: built from the CIE xy chromaticities of both primaries sets and a Bradford chromatic
  adaptation from D65 (Rec.709 white) to the ACES white point, the usual construction (ACES TB-2014-004 notes).

The official LTX-2.5 SDR-to-HDR graph converts with ComfyUI core's ImageColorSpace (sRGB -> HDR ACEScct). Core has
no ACEScg input, so ACEScg sources use `acescg_to_acescct` here; tests compare our Rec.709 path against core's.
"""
from __future__ import annotations

import torch

# --- ACEScct (S-2016-001) -------------------------------------------------------------------------------------------
X_BRK = 0.0078125                   # 2**-7: linear break point
Y_BRK = 0.155251141552511           # ACEScct value at the break point
A = 10.5402377416545                # slope of the linear segment
B = 0.0729055341958355              # offset of the linear segment


def acescct_encode(lin: torch.Tensor) -> torch.Tensor:
    """Linear AP1 (ACEScg) -> ACEScct. Extended (> 1) and negative values stay defined (the linear toe)."""
    x = lin.to(torch.float32)
    log = (torch.log2(torch.clamp(x, min=X_BRK)) + 9.72) / 17.52
    return torch.where(x <= X_BRK, A * x + B, log)


def acescct_decode(cct: torch.Tensor) -> torch.Tensor:
    """ACEScct -> linear AP1 (ACEScg); the exact inverse of acescct_encode."""
    y = cct.to(torch.float32)
    exp = torch.pow(2.0, torch.clamp(y, max=15.0) * 17.52 - 9.72)
    return torch.where(y <= Y_BRK, (y - B) / A, exp)


def acescg_to_acescct(img: torch.Tensor) -> torch.Tensor:
    """IMAGE [..., 3(+alpha)] in ACEScg -> ACEScct; alpha (a 4th channel) passes through untouched."""
    rgb = acescct_encode(img[..., :3])
    return torch.cat([rgb, img[..., 3:].to(rgb.dtype)], dim=-1) if img.shape[-1] > 3 else rgb


# --- primaries -> matrix --------------------------------------------------------------------------------------------
REC709 = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060))
AP1 = ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044))
D65 = (0.3127, 0.3290)
ACES_WHITE = (0.32168, 0.33767)
BRADFORD = ((0.8951, 0.2664, -0.1614), (-0.7502, 1.7135, 0.0367), (0.0389, -0.0685, 1.0296))


def _xyz(xy):
    x, y = xy
    return torch.tensor([x / y, 1.0, (1.0 - x - y) / y], dtype=torch.float64)


def rgb_to_xyz(primaries, white) -> torch.Tensor:
    """3x3 matrix taking linear RGB with these primaries / white to CIE XYZ (Y of white = 1)."""
    m = torch.stack([_xyz(p) for p in primaries], dim=1)
    s = torch.linalg.solve(m, _xyz(white))
    return m * s[None, :]


def bradford(src_white, dst_white) -> torch.Tensor:
    b = torch.tensor(BRADFORD, dtype=torch.float64)
    ratio = (b @ _xyz(dst_white)) / (b @ _xyz(src_white))
    return torch.linalg.inv(b) @ torch.diag(ratio) @ b


def rec709_to_ap1_matrix() -> torch.Tensor:
    """Linear Rec.709 (D65) -> linear AP1 (ACES white), Bradford-adapted."""
    return (torch.linalg.inv(rgb_to_xyz(AP1, ACES_WHITE)) @ bradford(D65, ACES_WHITE)
            @ rgb_to_xyz(REC709, D65))


def apply_matrix(img: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    rgb = img[..., :3].to(torch.float32) @ m.to(torch.float32).T
    return torch.cat([rgb, img[..., 3:].to(rgb.dtype)], dim=-1) if img.shape[-1] > 3 else rgb


INPUT_SPACES = ("srgb_gamma", "srgb", "acescg", "acescct")
# what core's ImageColorSpace calls the two spaces it converts for us
CORE_SOURCE_NAME = {"srgb_gamma": "sRGB", "srgb": "linear"}
