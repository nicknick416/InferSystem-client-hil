"""Hardware-free translation-stall and integration regression tests."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from HIL.stuck_detection import StuckDetector
from HIL.runtime import check_config
from HIL.recording import Episode, compact_records, records

ROOT = Path(__file__).resolve().parents[1]

def feed(detector, count=40, start=0, measured=lambda i: [0, 0, 0],
         target=lambda i: [.01, 0, 0], **kwargs):
    return [detector.update(start+i/30, measured(i), target(i), **kwargs) for i in range(count)]

def test_disabled_is_noop_and_does_not_validate_observations():
    d = StuckDetector()
    assert d.update(None, None, None) is None
    assert not d.samples

def test_sustained_stall_enters_once_then_clears_at_target():
    d = StuckDetector({"enabled": True})
    rows = feed(d)
    assert sum(r["event"] == "entered" for r in rows) == 1
    first = next(i for i, r in enumerate(rows) if r["stuck"])
    assert first/30 >= .75
    assert rows[-1]["stuck"]
    r = d.update(40/30, [.01, 0, 0], [.01, 0, 0])
    assert r["event"] == "cleared" and not r["stuck"]

def test_normal_motion_and_oscillation_are_not_stalls():
    for pos in (lambda i: [i*.0002, 0, 0],
                lambda i: [.002*np.sin(i*.6), 0, 0]):
        d = StuckDetector({"enabled": True})
        rows = feed(d, count=90, measured=pos, target=lambda i: [.04, 0, 0])
        assert not any(r["stuck"] for r in rows)

def test_human_and_resume_reset_evidence():
    d = StuckDetector({"enabled": True})
    assert feed(d)[-1]["stuck"]
    r = d.update(40/30, [0]*3, [.01, 0, 0], mode="human")
    assert r["event"] == "cleared"
    assert not any(r["stuck"] for r in feed(d, count=15, start=41/30))

def test_stale_network_hold_expires_and_empty_queue_never_arms():
    d = StuckDetector({"enabled": True})
    assert not any(r["stuck"] for r in feed(d, fresh_command=False))
    d.update(2, [0]*3, [.01, 0, 0], fresh_command=True)
    rows = feed(d, count=45, start=2+1/30, fresh_command=False)
    assert rows[-1]["reason"] == "stale_command"
    assert not rows[-1]["stuck"]

@pytest.mark.parametrize("timestamp", [5., .1])
def test_sample_gap_and_time_reversal_clear(timestamp):
    d = StuckDetector({"enabled": True})
    assert feed(d)[-1]["stuck"]
    r = d.update(timestamp, [0]*3, [.01, 0, 0])
    assert r["reason"] == "sample_gap" and r["event"] == "cleared"

def test_reversed_command_clears_and_inputs_not_mutated():
    d = StuckDetector({"enabled": True})
    feed(d)
    state = np.zeros(3)
    target = np.array([-.01, 0, 0])
    r = d.update(40/30, state, target)
    assert not r["stuck"] and r["event"] == "cleared"
    np.testing.assert_array_equal(state, [0, 0, 0])
    np.testing.assert_array_equal(target, [-.01, 0, 0])

def test_invalid_sample_clears_without_poisoning_next_episode():
    d = StuckDetector({"enabled": True})
    feed(d)
    r = d.update(float("nan"), [0]*3, [.01, 0, 0])
    assert r["reason"] == "invalid_sample" and not r["stuck"]
    json.dumps(r, allow_nan=False)
    assert feed(d)[-1]["stuck"]
    d.reset()
    assert not d.stuck and not d.samples

@pytest.mark.parametrize("config", [
    {"enabled": "false"}, {"window_s": 0}, {"confirm_s": float("nan")},
    {"min_samples": 1}, {"min_samples": True}, {"command_timeout_s": .1},
    {"max_position_excursion_m": .004}, {"min_direction_cosine": 2},
    {"typo": True}, [],
])
def test_invalid_config_fails_early(config):
    with pytest.raises(ValueError):
        StuckDetector(config)

def test_yaml_default_and_enabled_check_only(tmp_path):
    path = ROOT/"Config/flexiv_vtla_expo_hil.yaml"
    _, raw = check_config(path)
    assert StuckDetector().enabled is False
    raw["hil"]["stuck_detection"]["enabled"] = True
    enabled = tmp_path/"enabled.yaml"
    enabled.write_text(yaml.safe_dump(raw))
    check_config(enabled)
    raw["hil"].pop("stuck_detection")
    enabled.write_text(yaml.safe_dump(raw))
    if raw["hil"].get("stuck_recovery", {}).get("enabled"):
        with pytest.raises(ValueError, match="requires stuck_detection"):
            check_config(enabled)
    raw["hil"].pop("stuck_recovery", None)
    raw["hil"].pop("correction", None)
    enabled.write_text(yaml.safe_dump(raw))
    check_config(enabled)

def test_recording_accepts_event_without_changing_compact_control_stream(tmp_path):
    e = Episode(tmp_path, {"horizon": 50, "fps": 30})
    e.add("stuck_detection", tick=0, stuck=True, event="entered")
    e.add("tick", tick=0, stuck_detection={"stuck": True})
    e.seal()
    assert [r["kind"] for r in records(e.path)] == ["stuck_detection", "tick"]
    compact = list(compact_records(e.path))
    assert len(compact) == 1 and compact[0]["stuck_detection"]["stuck"]


def test_hysteresis_holds_between_thresholds_then_clears():
    d = StuckDetector({"enabled": True, "min_tracking_error_m": .0015,
                       "release_tracking_error_m": .00075})
    assert feed(d)[-1]["stuck"]
    for i in range(40, 70):
        result = d.update(i/30, [0, 0, 0], [.001, 0, 0])
        assert result["stuck"] and result["event"] is None
    result = d.update(70/30, [0, 0, 0], [.0005, 0, 0])
    assert result["event"] == "cleared"
    assert result["reason"] == "below_error_threshold"
    assert not any(r["stuck"] for r in feed(d, start=71/30, target=lambda i: [.001, 0, 0]))

def test_hysteresis_still_clears_on_takeover_and_movement():
    for mode in ("human", "policy"):
        d = StuckDetector({"enabled": True, "release_tracking_error_m": .00075})
        assert feed(d)[-1]["stuck"]
        result = d.update(40/30, [.002, 0, 0], [.01, 0, 0], mode=mode)
        assert result["event"] == "cleared"
        assert result["reason"] == ("non_policy" if mode == "human" else "moving")

@pytest.mark.parametrize("value", [-1, 0, float("nan"), True, .004])
def test_invalid_release_threshold(value):
    with pytest.raises(ValueError):
        StuckDetector({"release_tracking_error_m": value})

def test_omitted_release_retains_single_threshold_behavior():
    d = StuckDetector({"enabled": True})
    assert d.config["release_tracking_error_m"] == .003
    assert feed(d)[-1]["stuck"]
    assert not d.update(40/30, [0,0,0], [.002,0,0])["stuck"]
