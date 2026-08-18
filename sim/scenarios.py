"""
Phase 6 — scenario diversity (blueprint 3, Phase 6):

    "Design 7 (or more) distinct 3-D multi-target simulation scenarios
    spanning different trajectory geometries (parallel, crossing,
    converging paths, etc.), mirroring Fig. 23 of the paper."

The extracted paper analysis this project is built from is text-only (no
Fig. 23 image), so the specific geometries below are this project's own
design, not a reproduction of the paper's exact scenarios — built to cover
the same *kinds* of geometry the blueprint names, plus a second axis the
blueprint doesn't explicitly call out but that Phase 2/5's verification
work showed actually matters: scenarios where the two sensors genuinely
*disagree* or degrade differently.

That second axis is why SCENARIOS below isn't just "4 trajectory shapes":
Phase 2's KL-divergence-optimal omega labels came out within 1e-8 of
exactly 0.5 across ~40 runs of the single benign Phase 1 scenario — both
sensors were unbiased and well-calibrated, so they always agreed, and CI's
own math already handles that case without omega needing to do anything
(see fusion/kl_objective.py's docstring). Scenarios 5-7 (the "*_degraded"
and "high_clutter_*" ones) were built to break that via elevated clutter —
and, on testing, didn't: clutter delays a track's *confirmation* but
doesn't bias a *confirmed* track, so the two sensors kept agreeing on the
mean regardless of how different their covariances were, and per the
finding above, agreeing means pin the KL objective to exactly 0.5 no
matter what. That result is what motivated scenario 8, "ground_biased":
a *persistent measurement bias* (via sim.sensors.apply_measurement_bias),
not more clutter — a genuinely different lever, since bias moves the
*mean*, which is the one thing clutter/geometry alone couldn't touch.

Geometries were range-checked numerically (see git history / session notes)
against both sensors' max_range=3000m before being hardcoded here — the
same process used to tune Phase 1's original scenario.
"""
from dataclasses import dataclass


@dataclass
class ScenarioConfig:
    name: str
    description: str
    target_initial_states: list  # list of [x, vx, y, vy, z, vz], one per target
    ground_radar_position: list
    airborne_initial_state: list
    ground_clutter_rate: float = 1.0
    airborne_clutter_rate: float = 1.0
    num_steps: int = 14
    # [elevation, bearing, range] constant offset applied to real detections
    # (not clutter) — see sim.sensors.apply_measurement_bias. None = no bias.
    ground_measurement_bias: list = None
    airborne_measurement_bias: list = None


# Phase 1's original scenario, as a ScenarioConfig — same geometry as
# sim.two_radar_simulation's DEFAULT_* constants (kept there too, and
# run_two_radar_simulation keeps using those directly rather than this
# object, for zero risk to already-verified Phase 1-5 code). This exists so
# Phase 6's comparison pipeline can include "the clean, undegraded
# condition" as one point of comparison alongside the diversity scenarios
# below, using the same ScenarioConfig-based machinery.
BASELINE_SCENARIO = ScenarioConfig(
    name="baseline",
    description="Phase 1's original scenario: single target, no clutter elevation, no bias.",
    target_initial_states=[[0, 50, 0, 50, 1000, 0]],
    ground_radar_position=[-2000, 0, 0],
    airborne_initial_state=[1500, -30, 1500, -30, 2500, -20],
    num_steps=16,
)


# Shared airborne platform geometry for the new multi-target scenarios
# (1-4, 7) — re-verified in range for these specific target paths; the
# original Phase 1 scenario (used unchanged by scenarios 5-6, and by every
# Phase 1-5 module via sim.two_radar_simulation.DEFAULT_*) keeps its own
# separately-tuned geometry rather than being folded into this one.
_GROUND_POS = [-1500, -200, 0]
_AIRBORNE_INIT = [1200, -25, 1200, -25, 2200, -15]

SCENARIOS = [
    ScenarioConfig(
        name="parallel",
        description="Two targets on parallel, non-crossing straight-line paths.",
        target_initial_states=[
            [0, 45, 0, 45, 1000, 0],
            [300, 45, -300, 45, 1000, 0],
        ],
        ground_radar_position=_GROUND_POS,
        airborne_initial_state=_AIRBORNE_INIT,
    ),
    ScenarioConfig(
        name="crossing",
        description="Two targets on paths that cross near the scenario center "
                    "(closest approach ~t=7s) — stresses JPDA/GM-LCC data association.",
        target_initial_states=[
            [-350, 50, -350, 50, 1000, 0],
            [350, -50, -350, 50, 1000, 0],
        ],
        ground_radar_position=_GROUND_POS,
        airborne_initial_state=_AIRBORNE_INIT,
    ),
    ScenarioConfig(
        name="converging",
        description="Two targets starting far apart, moving toward a shared region "
                    "without exactly crossing.",
        target_initial_states=[
            [-600, 50, -600, 45, 1000, 0],
            [600, -45, -500, 40, 1000, 0],
        ],
        ground_radar_position=_GROUND_POS,
        airborne_initial_state=_AIRBORNE_INIT,
    ),
    ScenarioConfig(
        name="diverging",
        description="Two targets starting close together, spreading apart "
                    "over the run — the reverse of 'converging'.",
        target_initial_states=[
            [0, 55, 0, 15, 1000, 0],
            [50, -15, 50, 55, 1000, 0],
        ],
        ground_radar_position=_GROUND_POS,
        airborne_initial_state=_AIRBORNE_INIT,
    ),
    ScenarioConfig(
        name="airborne_degraded",
        description="Single target (Phase 1's baseline geometry, unchanged), but the "
                    "airborne radar's clutter rate is raised well above normal — the "
                    "ground radar should end up the more reliable sensor.",
        target_initial_states=[[0, 50, 0, 50, 1000, 0]],
        ground_radar_position=[-2000, 0, 0],
        airborne_initial_state=[1500, -30, 1500, -30, 2500, -20],
        ground_clutter_rate=1.0,
        airborne_clutter_rate=6.0,
        num_steps=16,
    ),
    ScenarioConfig(
        name="ground_degraded",
        description="Mirror of 'airborne_degraded': ground radar's clutter rate is "
                    "raised, airborne stays clean.",
        target_initial_states=[[0, 50, 0, 50, 1000, 0]],
        ground_radar_position=[-2000, 0, 0],
        airborne_initial_state=[1500, -30, 1500, -30, 2500, -20],
        ground_clutter_rate=6.0,
        airborne_clutter_rate=1.0,
        num_steps=16,
    ),
    ScenarioConfig(
        name="high_clutter_multitarget",
        description="Two targets (the 'parallel' geometry) with both sensors' clutter "
                    "rates well above normal — stresses data association and sensor "
                    "reliability simultaneously.",
        target_initial_states=[
            [0, 45, 0, 45, 1000, 0],
            [300, 45, -300, 45, 1000, 0],
        ],
        ground_radar_position=_GROUND_POS,
        airborne_initial_state=_AIRBORNE_INIT,
        ground_clutter_rate=4.0,
        airborne_clutter_rate=4.0,
    ),
    ScenarioConfig(
        name="ground_biased",
        description="Single target, Phase 1's exact baseline geometry (isolates the "
                    "effect of bias from the geometry/clutter changes in every other "
                    "scenario here) — but the ground radar has a persistent +150m range "
                    "bias (10x its own range noise std of 15m, so this is a genuine "
                    "miscalibration, not just noise) while the airborne radar stays "
                    "clean. Built specifically to test whether *bias* (unlike clutter — "
                    "see 'airborne_degraded'/'ground_degraded' and fusion/kl_objective.py) "
                    "moves Approach 1's KL-optimal omega away from 0.5.",
        target_initial_states=[[0, 50, 0, 50, 1000, 0]],
        ground_radar_position=[-2000, 0, 0],
        airborne_initial_state=[1500, -30, 1500, -30, 2500, -20],
        ground_clutter_rate=1.0,
        airborne_clutter_rate=1.0,
        ground_measurement_bias=[0.0, 0.0, 150.0],
        num_steps=16,
    ),
]

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}
SCENARIOS_BY_NAME[BASELINE_SCENARIO.name] = BASELINE_SCENARIO

# The subset of scenarios usable by Phase 6's cross-approach comparison
# pipeline (eval/compare_approaches.py) as-is: single target only, since
# Approaches 1-4's per-step fusion assumes one track pair per sensor and
# multi-target scenarios would need cross-sensor track-to-track association
# first (see run_scenario_simulation's and run_single_target_scenario's
# docstrings — deliberately not attempted here).
SINGLE_TARGET_SCENARIOS = [
    BASELINE_SCENARIO,
    SCENARIOS_BY_NAME["airborne_degraded"],
    SCENARIOS_BY_NAME["ground_degraded"],
    SCENARIOS_BY_NAME["ground_biased"],
]
