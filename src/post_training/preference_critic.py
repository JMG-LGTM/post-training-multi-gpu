"""Preference-regularized twin Q while retaining PixelIQL's TD/V objectives.

This is an experimental IDQL critic extension, not FlowPRO's actor RPRO loss.
The actor stays outside this module. No network, export schema, sparse reward,
target update, value expectile, or inference selection rule is changed here.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext

import torch
from torch.nn import functional as F

from .critic import PixelIQL


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _validate_pairs(model, pairs):
    required = {"pixels", "proprio", "chosen_actions", "rejected_actions", "action_mask"}
    if not isinstance(pairs, Mapping) or not required <= pairs.keys():
        raise ValueError("Preference batch requires shared pixels/proprio, chosen/rejected actions and action_mask")
    values = {key: pairs[key] for key in required}
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError("Preference batch fields must be tensors")
    pixels, proprio = values["pixels"], values["proprio"]
    chosen, rejected, mask = (values[key] for key in ("chosen_actions", "rejected_actions", "action_mask"))
    if pixels.ndim not in (4, 5) or any(size < 1 for size in pixels.shape):
        raise ValueError("Preference pixels must be nonempty [P,C,H,W] or [P,cameras,3,H,W]")
    channels = pixels.shape[1] if pixels.ndim == 4 else pixels.shape[1] * pixels.shape[2]
    if (channels != model.model_config["pixel_channels"] or pixels.ndim == 5 and pixels.shape[2] != 3):
        raise ValueError("Preference pixel channels differ from the critic")
    size = pixels.shape[0]
    expected = (size, model.model_config["horizon"], model.model_config["action_dim"])
    if chosen.shape != expected or rejected.shape != expected:
        raise ValueError("Preference chosen/rejected action shapes differ from critic dimensions")
    if proprio.shape != (size, model.model_config["proprio_dim"]):
        raise ValueError("Preference proprio shape differs from critic dimensions")
    if mask.shape != expected[:2]:
        raise ValueError("Preference actions must share one [P,horizon] action mask")
    device = next(model.q1.parameters()).device
    if any(value.device != device for value in values.values()):
        raise ValueError("Preference tensors must be on the critic device")
    if not all(value.is_floating_point() for value in (proprio, chosen, rejected)):
        raise ValueError("Preference proprio/actions must be floating normalized tensors")
    if pixels.dtype != torch.uint8 and not pixels.is_floating_point():
        raise ValueError("Preference pixels must be uint8 or floating in the 0..255 domain")
    if mask.is_complex() or not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("Preference inputs must be finite, including padded actions")
    if bool((pixels < 0).any() or (pixels > 255).any()):
        raise ValueError("Preference pixels must remain in the 0..255 domain")
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool(mask.bool().any(1).all()):
        raise ValueError("Preference action mask must be binary and nonempty per pair")
    return pixels, proprio, chosen, rejected, mask

def preference_ranking(model: PixelIQL, pairs, *, temperature=1., target=False):
    """Rank pairs without native transitions, augmentation or mode/RNG changes.

    Online rankings retain gradients for the training objective. Target rankings
    explicitly disable gradients, matching the frozen target-Q interpretation.
    Call under no_grad for held-out online diagnostics as well.
    """
    temperature = _positive(temperature, "preference_temperature")
    if type(target) is not bool:
        raise ValueError("target must be boolean")
    pixels, proprio, chosen, rejected, mask = _validate_pairs(model, pairs)
    q1 = model.target_q1 if target else model.q1
    q2 = model.target_q2 if target else model.q2
    context = torch.no_grad() if target else nullcontext()
    with context:
        q1w, q1l = q1(pixels, proprio, chosen, mask), q1(pixels, proprio, rejected, mask)
        q2w, q2l = q2(pixels, proprio, chosen, mask), q2(pixels, proprio, rejected, mask)
        margins = (q1w - q1l, q2w - q2l)
        if not all(bool(torch.isfinite(value).all()) for value in (*margins, q1w, q1l, q2w, q2l)):
            raise ValueError("Preference Q scores or margins are nonfinite")
        loss = .5 * (F.softplus(-margins[0] / temperature) + F.softplus(-margins[1] / temperature))
        if not bool(torch.isfinite(loss).all()):
            raise ValueError("Preference loss overflowed; check Q scale and temperature")
        min_margin = torch.minimum(q1w, q2w) - torch.minimum(q1l, q2l)
        return {"pref_loss": loss, "preference_loss": loss,
                "pref_accuracy": .5 * ((margins[0] > 0).float() + (margins[1] > 0).float()),
                "pref_margin": (.5 * (margins[0] + margins[1])).detach(),
                "pref_q1_accuracy": (margins[0] > 0).float(),
                "pref_q2_accuracy": (margins[1] > 0).float(),
                "pref_min_accuracy": (min_margin > 0).float(),
                "pref_min_margin": min_margin.detach()}


class PreferenceIQL(PixelIQL):
    """Add headwise Bradley--Terry rankings to online Q1/Q2 only.

    ``preference_batch`` supplies one shared ``pixels``/``proprio`` observation,
    ``chosen_actions``, ``rejected_actions`` and one common ``action_mask``.
    The trainer normalizes the raw actions/state and augments each pair's image
    once before this call. This module performs no augmentation or RNG draws.

    q_loss[B] = native_TD_loss[B] + weight * mean(pair_loss[P]). Broadcasting the
    scalar preserves that coefficient under the caller's ordinary batch mean,
    including when B != P. ``pref_loss`` remains [P] for logging; do not add it
    a second time. Accuracy/margin average the two online heads. Additional
    min-Q diagnostics correspond to the online selection score, not target Q.

    Disabled preference (weight zero or no pairs), and the value-only stage,
    delegate exactly to PixelIQL, including its return keys and RNG behavior.
    The inherited constructor/state_dict/model_config remain exactly unchanged.
    """

    def forward(self, batch, stage="both", *, beta=3., max_weight=20.,
                preference_batch=None, preference_weight=.1, preference_temperature=1.):
        if stage == "value" or preference_batch is None or preference_weight == 0:
            return super().forward(batch, stage=stage, beta=beta, max_weight=max_weight)
        weight = _positive(preference_weight, "preference_weight")
        temperature = _positive(preference_temperature, "preference_temperature")
        result = super().forward(batch, stage=stage, beta=beta, max_weight=max_weight)
        ranking = preference_ranking(self, preference_batch, temperature=temperature)
        td_loss = result["q_loss"]
        result.update(ranking)
        result.update(q_td_loss=td_loss, q_loss=td_loss + weight * ranking["pref_loss"].mean())
        return result
