# Preference supervision for a frozen-policy IDQL scorer

This extension keeps the BC behavior policy fixed and adds local human preference
supervision to the existing twin-Q learner. It is an experimental modification
of the critic, **not FlowPRO/RPRO**. RPRO fine-tunes a flow policy against a frozen
reference policy; no RPRO actor objective is used here.

## Objective and inference

The native transition updates remain V expectile regression, then twin-Q Bellman
regression, then target-Q EMA. The reward, discount and termination rules remain
the existing sparse-reward contract. A preferred and rejected action chunk must
share the same conditioning state and valid horizon.

For each Q head, the preference objective is

```text
delta_i = (Q_i(s, chosen) - Q_i(s, rejected)) / temperature
L_pref  = mean_pairs[0.5 * (softplus(-delta_1) + softplus(-delta_2))]
L_Q     = L_native_TD + preference_weight * L_pref
```

Both terms contribute to the **same Q optimizer step**. V has no separate
preference loss; it follows the updated target Q through its existing expectile
objective. The target networks remain frozen to gradients and follow EMA.
There is no actor optimizer, reward-model fitting, extra Q head or change to the
behavior actor. Preference accuracy and margins are diagnostics, not rollout success.

At inference, the unchanged frozen behavior policy supplies candidates. The
existing target `min(Q1,Q2)` score and argmax selection can consume the exported
critic because the architecture and input contract remain unchanged. A new
evaluation must bind the new critic SHA; the running evaluation continues to use
its original immutable critic. This repository change does not switch it.

## Data requirements

Success and failure trajectories cannot be paired by matching row numbers. Human
and policy trajectories generally diverge after intervention. A portable pair
contains a single conditioning observation/state, two action chunks, a shared
mask, root lineage and supplied same-state evidence. The package validates that
evidence and its file hashes; this does not independently establish that a human
preference label is correct. Actual source recordings still require an audit.

Derive the authoritative training/validation membership from the **complete
native transition dataset**, then assign each preference pair using its root.
An optional frozen-membership file can bind an existing experiment's exact
roster. Do not recalculate fixed-count splits on the smaller preference subset.
The default follows Haizhou's SHA256(seed:root) threshold rule, fraction 0.2,
seed 7; an explicitly configured alternative uses the native dataset's map.
Ancestors, branches and shared prefixes stay in the same split. No dataset size,
root count, experiment path or behavior-policy identity is an algorithm constant.

The first human action segment identifies the potential shared fork. A final
branch's `resume_step` can instead identify the later handback to the policy.
Further human states require independently aligned alternatives; they are not
automatically additional pairs. Smooth pose interpolation from FlowPRO cannot
be copied directly into LIBERO control increments.

Only training preference pairs enter optimization. Native validation and
preference validation are separate; their counts and root members must be
reported. A very small preference holdout cannot establish generalization.
Demonstrations with no losing alternatives are not fabricated into strict
preference labels. The portable data adapter currently follows the existing
runner's pi0.5 LIBERO-X V1 I/O contract (two cameras, 8-step action chunks,
7 actions and 8 proprioception values). Supporting another robot's raw I/O
requires a matching adapter; the preference critic itself uses its configured
model dimensions.

## Controlled comparison

Compare two independently named runs with identical native members,
normalization, initial Q/V weights or seed, update budget, native batch size,
optimizer settings and evaluation protocol:

1. Preference weight 0: original critic update, no preference sampling or extra
   random draws.
2. Positive preference weight: the same update with the auxiliary ranking term.

The native batch and the additional preference-pair batch are different budgets
and must be reported separately. Repeated draws from a small pair pool are
explicit replacement sampling, not new human examples. Neither variant is
guaranteed to exceed its behavior policy or the unmodified IDQL scorer.

Checkpoints bind the native data/cache identity, preference package, frozen root
membership, behavior-policy identity, configuration and source hashes. A changed
preference weight or changed pair package requires a separate experiment; it is
not an exact resume of the old run. Frozen BC weights are not duplicated into
critic checkpoints.

## Sources and status

- [IDQL paper](https://arxiv.org/abs/2304.10573)
- [FlowPRO/RPRO paper](https://arxiv.org/html/2606.05468v1)
- [FlowPRO project](https://wuyeyexvnainai.github.io/flowpro/)

The original FlowPRO project currently advertises its training-code release as
coming soon. This preference-regularized Q objective is our implementation
choice; it must not be described as reproducing official RPRO code or its
reported success rates. Numerical, CPU DDP and resume checks are recorded
separately from real CUDA training and rollout evaluations.
