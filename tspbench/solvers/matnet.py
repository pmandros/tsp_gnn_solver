"""MatNet (Kwon et al., NeurIPS 2021), run from its released code and ATSP checkpoints.

MatNet is the standard learned baseline for matrix-input routing: it reads the
full (possibly asymmetric) cost matrix, so it is the natural comparison for
non-metric and asymmetric instances. The code is not vendored. Fetch it with

    python scripts/download_benchmarks.py --matnet tools/MatNet
    export MATNET_DIR=$PWD/tools/MatNet

which clones github.com/yd-kwon/MatNet at a pinned commit (MIT licence). The
repository ships checkpoints trained on ATSP "tmat" instances with n = 20, 50
and 100 (``atsp{n}`` in ``tspbench``).

Spec: ``matnet:model=100,aug=1,decode=argmax``

* ``model``: which checkpoint, 20 / 50 / 100. Default: the smallest one with
  ``model >= n``, else 100.
* ``aug``: MatNet's instance augmentation, i.e. that many random draws of the
  one-hot column embeddings, keeping the best tour. The paper reports 1 and 128.
* ``decode``: ``argmax`` (greedy POMO rollouts from every start node) or
  ``softmax`` (sampled rollouts, what the released ``test.py`` uses).
* ``scale``: ``none`` (default), ``max``, which divides the matrix by its
  largest entry first (use it for integer-cost files such as TSPLIB), or a
  number to multiply the matrix by.

The column embeddings are one-hot vectors in the 256-d embedding space, so the
architecture cannot take n > 256; larger instances raise ``Unsupported``.
By default costs are passed as they are: MatNet was trained on matrices in [0, 1], which
is also the range of the ``atsp{n}`` and ``nonmetric{n}`` generators.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

import numpy as np

from .base import Solver, SolverUnavailable, Unsupported, register

CHECKPOINTS = {
    20: "result/saved atsp20_model/checkpoint-5000.pt",
    50: "result/saved_atsp50_model/checkpoint-8000.pt",
    100: "result/saved_atsp100_model/checkpoint-12000.pt",
}
EMBEDDING_DIM = 256

# Hyperparameters from the released ATSP_MatNet/test.py (identical for all three checkpoints).
MODEL_PARAMS = {
    "embedding_dim": EMBEDDING_DIM,
    "sqrt_embedding_dim": EMBEDDING_DIM ** 0.5,
    "encoder_layer_num": 5,
    "qkv_dim": 16,
    "sqrt_qkv_dim": 16 ** 0.5,
    "head_num": 16,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "ms_hidden_dim": 16,
    "ms_layer1_init": (1 / 2) ** 0.5,
    "ms_layer2_init": (1 / 16) ** 0.5,
}


def _matnet_src():
    root = os.environ.get("MATNET_DIR")
    if not root:
        return None
    src = os.path.join(root, "ATSP", "ATSP_MatNet")
    return src if os.path.isfile(os.path.join(src, "ATSPModel.py")) else None


@lru_cache(maxsize=None)
def _load(size: int):
    import torch

    src = _matnet_src()
    if src not in sys.path:
        sys.path.insert(0, src)
    from ATSPModel import ATSPModel

    model = ATSPModel(**MODEL_PARAMS, eval_type="argmax", one_hot_seed_cnt=size)
    ck = torch.load(os.path.join(src, CHECKPOINTS[size]), map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    return model.eval()


def rollout(model, problems, decode="argmax"):
    """POMO rollouts from every start node. ``problems``: (batch, n, n) float tensor.

    Returns tours (batch, n, n) and their lengths (batch, n): one tour per start node.
    """
    import torch

    from ATSPModel import _get_encoding

    b, n, _ = problems.shape
    model.model_params["eval_type"] = decode
    bidx = torch.arange(b)[:, None].expand(b, n)
    pidx = torch.arange(n)[None, :].expand(b, n)
    with torch.no_grad():
        model.pre_forward(_Reset(problems))
        cur = torch.arange(n)[None, :].expand(b, n)
        model.decoder.set_q1(_get_encoding(model.encoded_row, cur))
        mask = torch.zeros(b, n, n)
        mask[bidx, pidx, cur] = float("-inf")
        tours = [cur]
        for _ in range(n - 1):
            probs = model.decoder(_get_encoding(model.encoded_row, cur), ninf_mask=mask)
            if decode == "argmax":
                cur = probs.argmax(dim=2)
            else:
                while True:  # multinomial can pick zero-probability entries; the released code retries too
                    cur = probs.reshape(b * n, n).multinomial(1).reshape(b, n)
                    if (probs[bidx, pidx, cur] > 0).all():
                        break
            mask[bidx, pidx, cur] = float("-inf")
            tours.append(cur)
    tours = torch.stack(tours, 2)
    nxt = tours.roll(-1, dims=2)
    lengths = problems[torch.arange(b)[:, None, None], tours, nxt].sum(2)
    return tours, lengths


class _Reset:
    def __init__(self, problems):
        self.problems = problems


@register("matnet")
class MatNet(Solver):
    stochastic = True

    @classmethod
    def available(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "pip install torch"
        if _matnet_src() is None:
            return False, "run scripts/download_benchmarks.py --matnet tools/MatNet and set MATNET_DIR"
        return True, ""

    def solve(self, inst, seed=0):
        import torch

        n = inst.n
        if n > EMBEDDING_DIM:
            raise Unsupported(f"MatNet's one-hot embeddings allow n <= {EMBEDDING_DIM}")
        size = int(self.params.get("model", next((s for s in sorted(CHECKPOINTS) if s >= n), max(CHECKPOINTS))))
        if size not in CHECKPOINTS:
            raise SolverUnavailable(f"no MatNet checkpoint for model={size}; have {sorted(CHECKPOINTS)}")
        model = _load(size)
        # The one-hot pool must hold n distinct codes; beyond the training size this is extrapolation.
        model.model_params["one_hot_seed_cnt"] = max(size, n)
        aug = int(self.params.get("aug", 1))
        batch = int(self.params.get("batch", 16))
        d = np.asarray(inst.full_matrix(), dtype=np.float64)
        scale = self.params.get("scale", "none")
        if scale == "max":
            d = d / d.max()
        elif scale != "none":
            d = d * float(scale)
        d = torch.as_tensor(d, dtype=torch.float32)
        torch.manual_seed(seed)
        best_len, best_tour = float("inf"), None
        for start in range(0, aug, batch):
            k = min(batch, aug - start)
            tours, lengths = rollout(model, d[None].expand(k, n, n).contiguous(), self.params.get("decode", "argmax"))
            i = int(lengths.argmin())
            if float(lengths.view(-1)[i]) < best_len:
                best_len = float(lengths.view(-1)[i])
                best_tour = tours.view(-1, n)[i]
        return best_tour.numpy().astype(np.int64)
