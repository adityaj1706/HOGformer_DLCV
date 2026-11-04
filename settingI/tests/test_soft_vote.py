"""Sanity tests for the soft-voting HOG descriptor (run from settingI/):
    PYTHONPATH=. python tests/test_soft_vote.py
They check the two claims used in the report's justification:
  (1) the histogram is continuous in the gradient orientation with soft voting (it jumps with hard binning);
  (2) gradient information reaches the orientation with soft voting (it is exactly zero with hard binning).
"""
import math, sys, torch
sys.path.insert(0, 'basicsr')
from models.archs.hogformer_arch import Attention_DHOGSA, HOGformer

torch.manual_seed(0)
att = Attention_DHOGSA(dim=8, num_heads=1, bias=False)
PATCH = 5  # an interior 8x8 patch of a 32x32 map (index row 1, col 1), away from the zero-padded border

def ramp(phi, size=32, c=4):
    """Image whose Sobel gradient has constant magnitude and orientation phi everywhere in the interior."""
    ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing='ij')
    img = 0.1 * (torch.cos(phi) * xs + torch.sin(phi) * ys)
    return img.expand(1, c, size, size)

def hist(phi, soft):
    att.soft_vote = soft
    return att.apply_hog_to_patch(ramp(phi))[2][0, PATCH]

# (1) continuity: sweep the orientation finely and look at the biggest change between neighbouring steps
phis = torch.linspace(-math.pi + 1e-3, math.pi - 1e-3, 3600)
for soft in (False, True):
    hs = torch.stack([hist(p, soft) for p in phis])
    step = (hs[1:] - hs[:-1]).abs().sum(-1).max().item()
    print(f"soft_vote={soft!s:5}: largest L1 change of the histogram between neighbouring angles = {step:.4f}")
    if soft: assert step < 0.05, step
    else:    assert step > 1.5, step
    assert torch.allclose(hs.sum(-1), torch.ones(len(phis)), atol=1e-4)   # normalised histograms

# (2) gradient w.r.t. the orientation: magnitude is constant in phi here, so any gradient is via the orientation
w = torch.arange(9, dtype=torch.float32)
phi0 = 0.37   # not at a bin centre
for soft in (False, True):
    phi = torch.tensor(phi0, requires_grad=True)
    (hist(phi, soft) * w).sum().backward()
    g = abs(phi.grad.item())
    print(f"soft_vote={soft!s:5}: |d loss / d orientation| = {g:.6f}")
    if soft: assert g > 1e-2, g
    else:    assert g < 1e-4, g

# (3) no extra parameters, finite outputs
kw = dict(inp_channels=3, out_channels=3, dim=12, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1,
          heads=[1, 2, 4, 8], ffn_expansion_factor=2.667, bias=False, LayerNorm_type='WithBias')
a, b = HOGformer(**kw), HOGformer(soft_vote=True, **kw)
assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
y = b(torch.rand(1, 3, 64, 64)); assert torch.isfinite(y).all()
print("ALL TESTS PASSED")
