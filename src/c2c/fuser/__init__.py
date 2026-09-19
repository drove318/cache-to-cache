"""The cache fuser — the only trainable part of a C2C deployment.

Paper map (normative, spec §1):

* §3.3.1 Fused-cache update, Eqs. (3)–(4) → :mod:`c2c.fuser.core`
* §3.3.2 Projection, dynamic weighting, gate (Fig. 5) → :mod:`c2c.fuser.modules`
* App. A.1.3 C2C-C complex variant (Table 9) → :mod:`c2c.fuser.complex`
* App. A.2.4 Progressive behaviour (Fig. 11) → :mod:`c2c.fuser.blend`

Every class here is an ``nn.Module`` clone: the fuser holds one pair module
per mapped layer pair, exactly as the paper prescribes. Both communicating
models stay frozen at all times (FR-02); only these modules learn.
"""

from __future__ import annotations

from .blend import apply as apply_blend
from .blend import sweep
from .complex import PreProjection
from .core import FUSER_VARIANTS, Fuser, FuserPair
from .modules import DynamicWeighting, Gate, Projection

__all__ = [
    "Fuser", "FuserPair", "Projection", "DynamicWeighting", "Gate",
    "PreProjection", "apply_blend", "sweep", "FUSER_VARIANTS",
]
