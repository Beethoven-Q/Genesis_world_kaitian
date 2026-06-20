# SPDX-License-Identifier: Apache-2.0
"""The DR (domain-randomization) package — agent-native tools for the DR strategist.

MVP scope (docs/agents.md "DR strategist"):
  - ``sweep`` : the explore->measure->record PROBE. Runs the existing collector on a SMALL batch for a
                candidate DR setting (range multipliers via env vars) and reports success rate + an achieved-
                diversity measure + where failures cluster, WITHOUT a full collection.

The actual DR RANGES still live (hand-written) in ``tasks/pickplace.py::sample_phys_dr`` + the stage; this
package does NOT own or mutate them. The strategist subagent ADVISES range edits; the main agent applies them.
A future fully-declarative ``dr/`` config (scopes.py / object_dr.py / sampler.py / apply.py / plan.py, per
docs/domain_randomization.md "How it's applied") would replace the in-line ranges — that is a larger refactor,
deliberately deferred past this MVP.
"""
