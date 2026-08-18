"""
Per-sensor GM-LCC tracker — Phase 1 of the blueprint (sensor 2 / ground radar):

    "Trackers compared per sensor: JPDA (airborne radar) and GM-LCC
    (ground radar)"

    Risk flagged in the blueprint (section 4.6): "Stone Soup may not ship
    [GM-LCC / GM-PHD] out of the box — budget extra time to implement or
    adapt ... or substitute a well-supported equivalent."

That risk turned out to be a non-issue: the installed Stone Soup 1.9.1 ships
``stonesoup.updater.pointprocess.LCCUpdater`` directly — a Gaussian-Mixture
Linear-Complexity-with-Cumulants filter citing the same D. E. Clark & F. De
Melo (2018) reference the paper itself cites for GM-LCC. No substitution
needed.

Stone Soup's own driver for this filter, ``PointProcessMultiTargetTracker``,
is generator/``DetectionReader``-based; this class inlines the same update
logic (see its ``__next__``) behind a step-by-step API instead, to match
JPDATracker and stay driven by the same manual simulation loop.
"""
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.updater.pointprocess import LCCUpdater
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track

DEFAULT_BIRTH_STATE = [0, 0, 0, 0, 0, 0]
DEFAULT_BIRTH_COVAR = [
    [1e6, 0, 0, 0, 0, 0],
    [0, 1e2, 0, 0, 0, 0],
    [0, 0, 1e6, 0, 0, 0],
    [0, 0, 0, 1e2, 0, 0],
    [0, 0, 0, 0, 1e6, 0],
    [0, 0, 0, 0, 0, 1e2],
]


class GMLCCTracker:
    """Wraps Stone Soup's GM-LCC (Gaussian Mixture, Linear Complexity with
    Cumulants) point-process filter behind a simple ``step`` API.

    Parameters
    ----------
    transition_model : TransitionModel
        Same NCV model used to generate ground truth (sim.scenario).
    birth_state_vector, birth_covar : array-like
        Where/how uncertain new-target births are expected to appear. The
        wide default (variance 1e6 in position) means "anywhere in the
        scene", same spirit as JPDATracker's default prior.
    mean_births_per_step : float
        Expected number of new targets per time step (Poisson mean) — the
        birth component's weight. Kept low so the filter doesn't spuriously
        spawn tracks out of clutter alone.
    prob_detect, prob_survive : float
        Detection / survival probability (paper doesn't give values; JPDA
        tracker uses the same default prob_detect=0.9 for consistency).
    clutter_spatial_density : float
        Must (roughly) match the clutter actually configured on the sensor
        (see sim/sensors.py) — the filter uses it to weigh clutter vs.
        target-originated detections. Unlike JPDA (which hard-gates
        unlikely detections regardless of this value), GM-LCC has no gate
        (prob_gate=1 below) so it's fully sensitive to this being roughly
        right: too small and clutter gets treated as plausible target
        detections, spawning spurious tracks. The default here was tuned
        empirically against sim.sensors' default clutter_rate=1.0 /
        world_extent — re-tune if you change either.
    mean_false_alarms, variance_false_alarms : float
        First/second moments of the clutter count distribution — this is
        exactly the "cumulants" LCC uses (over PHD) to also track the
        *variance* of cardinality, not just its mean.
    extraction_threshold : float
        Mixture components with weight above this are reported as tracks.
    prune_threshold, merge_threshold, max_number_components :
        Passed straight to GaussianMixtureReducer to keep the mixture size
        bounded.
    """

    def __init__(
        self,
        transition_model,
        birth_state_vector=DEFAULT_BIRTH_STATE,
        birth_covar=DEFAULT_BIRTH_COVAR,
        mean_births_per_step=0.05,
        prob_detect=0.9,
        prob_survive=0.99,
        clutter_spatial_density=1e-4,
        mean_false_alarms=1.0,
        variance_false_alarms=1.0,
        extraction_threshold=0.5,
        prune_threshold=1e-9,
        merge_threshold=16,
        max_number_components=100,
    ):
        self.predictor = ExtendedKalmanPredictor(transition_model)
        # Single-target Kalman update used *inside* the mixture (one call per
        # component-detection pair) — distinct from LCCUpdater, which
        # combines those into the new weighted mixture.
        inner_updater = ExtendedKalmanUpdater(measurement_model=None)
        inner_hypothesiser = PDAHypothesiser(
            predictor=self.predictor,
            updater=inner_updater,
            clutter_spatial_density=clutter_spatial_density,
            prob_detect=prob_detect,
            # prob_gate=1 -> no hard gating: GM-style filters weigh every
            # component-detection pair by likelihood instead of discarding
            # ones outside a validation gate.
            prob_gate=1.0,
        )
        self.hypothesiser = GaussianMixtureHypothesiser(
            inner_hypothesiser, order_by_detection=True)
        self.updater = LCCUpdater(
            updater=inner_updater,
            clutter_spatial_density=clutter_spatial_density,
            prob_detection=prob_detect,
            prob_survival=prob_survive,
            mean_number_of_false_alarms=mean_false_alarms,
            variance_of_false_alarms=variance_false_alarms,
        )
        self.reducer = GaussianMixtureReducer(
            prune_threshold=prune_threshold,
            merge_threshold=merge_threshold,
            max_number_components=max_number_components,
        )
        self.birth_component = TaggedWeightedGaussianState(
            tag=TaggedWeightedGaussianState.BIRTH,
            weight=mean_births_per_step,
            state_vector=birth_state_vector,
            covar=birth_covar,
            timestamp=None,
        )
        self.extraction_threshold = extraction_threshold
        self.gaussian_mixture = GaussianMixture()
        # target_tracks: only tags *currently* present in the mixture (used
        # to decide whether to extend vs. create a track, and drives
        # end-of-track bookkeeping). all_tracks: every track ever created,
        # kept even after its tag drops out of the mixture — a track dying
        # one step (e.g. a single missed detection) shouldn't erase its
        # entire prior history from evaluation/reporting, the way Stone
        # Soup's own PointProcessMultiTargetTracker.tracks would.
        self.target_tracks = dict()
        self.all_tracks = set()

    @property
    def tracks(self):
        """Every track ever created (for evaluation/reporting — SIAP/OSPA
        want a track's full history, not just what's alive this instant)."""
        return self.all_tracks

    @property
    def active_tracks(self):
        """Only tracks whose tag is present in the mixture *this* step (for
        real-time use, e.g. picking the current estimate to feed into CI
        fusion — a track that ended 5 steps ago shouldn't win "most recent
        estimate" just because it happens to be longer)."""
        return set(self.target_tracks.values())

    def step(self, detections, timestamp):
        """Advance the mixture by one time step given this step's detections.

        Returns ``self.tracks`` — every track ever created, including ones
        no longer active this step (also updated in place).
        """
        self.birth_component.timestamp = timestamp
        self.gaussian_mixture.append(self.birth_component)

        hypotheses = self.hypothesiser.hypothesise(
            self.gaussian_mixture.components, detections, timestamp)
        try:
            self.gaussian_mixture = self.updater.update(hypotheses)
            self.gaussian_mixture.components = self.reducer.reduce(self.gaussian_mixture.components)
        except ValueError:
            # LCCUpdater's cumulant correction terms (second_order_cumulant,
            # the Panjer-process alpha) can occasionally underflow to a
            # non-positive weight, which stonesoup.types.numeric.Probability
            # raises on (it represents probabilities in log-space and
            # log(<=0) is undefined) — a known numerical fragility of
            # GM-LCC/PHD-style filters, not a bug in this wiring. Treat it
            # like "lost track this step": reset the mixture (and the
            # updater's own cumulant state, which is exactly what made this
            # step unstable) rather than letting one bad step crash an
            # entire batch of simulations — see Phase 2 verification notes.
            self.gaussian_mixture = GaussianMixture()
            self.updater.second_order_cumulant = 0

        self._update_tracks()
        self._end_tracks()
        return self.tracks

    def _update_tracks(self):
        for component in self.gaussian_mixture:
            tag = component.tag
            if tag == component.BIRTH:
                continue
            if tag in self.target_tracks:
                self.target_tracks[tag].states.append(component)
            elif component.weight > self.extraction_threshold:
                track = Track([component], id=tag)
                self.target_tracks[tag] = track
                self.all_tracks.add(track)

    def _end_tracks(self):
        component_tags = {component.tag for component in self.gaussian_mixture}
        for tag in self.target_tracks.keys() - component_tags:
            del self.target_tracks[tag]
