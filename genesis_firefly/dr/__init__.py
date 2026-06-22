# SPDX-License-Identifier: Apache-2.0
"""The DR (domain-randomization) package — the TASK-AGNOSTIC, REUSABLE DR harness (the declarative half of
docs/domain_randomization.md "How it's applied"). Any task reuses it: scope A (scene) + scope C (visual) are
AUTOMATIC for every task; scope B is per-object (the task names its fields).

Modules:
  - ``scopes``     : SCENE_DR (scope A) + VISUAL_DR (scope C) field definitions that apply to EVERY task
                     (the implemented Phase-1 fields + documented Phase-2 stubs).
  - ``object_dr``  : scope B keyed by object name (colour policy / mass / friction / pose; size is a Phase-2 stub).
  - ``sampler``    : ``TaskSpec`` + ``sample_*`` — split a task's DR into a per-build ``BuildDR`` + a batched
                     per-env ``EnvDR``, preserving the EXACT pre-refactor RNG call order (the parity lever).
  - ``apply``      : ``apply_build_dr`` (colours before build) + ``apply_env_dr`` (batched per-env setters).
  - ``plan``       : ``demo_dr_attrs`` — assemble the per-demo ``dr_*`` HDF5 trace attrs.
  - ``sweep``      : the DR strategist's explore->measure PROBE (runs the collector on a small batch).

A task NEVER writes DR logic — it names which scope-B fields apply (its ``TaskSpec``); scopes A and C are free.
"""
from dr.scopes import SCENE_DR, VISUAL_DR, DRField, PER_BUILD, PER_ENV  # noqa: F401
from dr.object_dr import ObjectDR, object_dr, classify_colour  # noqa: F401
from dr.sampler import (TaskSpec, BuildDR, EnvDR, sample_env_phys, sample_build_colours,  # noqa: F401
                        sample_post_build)
from dr import apply, plan  # noqa: F401
