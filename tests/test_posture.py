import pytest

from ergotwin.angles import Angles
from ergotwin.config import AngleConfig, TimingConfig
from ergotwin.posture import PostureMonitor, RegionTimer, describe

TIMING = TimingConfig(debounce_s=0.5, lost_grace_s=1.0, bad_hold_s=5.0, caution_hold_s=60.0,
                      cooldown_s=20.0, static_s=60.0, track_timeout_s=3.0)
SAFE, CAUTION, BAD = 0, 1, 2


def feed(timer, start, end, level, step=0.1):
    """Feed one level from `start` to `end` seconds. Returns the times at which a flag fired."""
    fired = []
    for i in range(round((end - start) / step) + 1):
        t = round(start + i * step, 3)
        if timer.update(t, level):
            fired.append(t)
    return fired


def test_timer_counts_time_in_a_posture():
    timer = RegionTimer(TIMING)
    feed(timer, 0, 3, BAD)
    assert timer.level == BAD
    assert timer.held(3.0) == pytest.approx(3.0)


def test_a_short_blip_does_not_restart_the_clock():
    timer = RegionTimer(TIMING)
    feed(timer, 0, 3, BAD)
    feed(timer, 3.1, 3.3, SAFE)  # shorter than the debounce time
    feed(timer, 3.4, 4, BAD)
    assert timer.level == BAD
    assert timer.held(4.0) == pytest.approx(4.0)


def test_a_real_change_counts_from_when_it_first_appeared():
    timer = RegionTimer(TIMING)
    feed(timer, 0, 3, BAD)
    feed(timer, 3.1, 5, SAFE)
    assert timer.level == SAFE
    assert timer.held(5.0) == pytest.approx(1.9)


def test_bad_posture_is_flagged_after_the_hold_time_then_again_after_the_cooldown():
    timer = RegionTimer(TIMING)
    fired = feed(timer, 0, 30, BAD)
    assert len(fired) == 2
    assert fired[0] == pytest.approx(5.0)
    assert fired[1] == pytest.approx(25.0, abs=0.11)
    assert timer.flag_count == 2


def test_a_brief_bad_posture_is_not_flagged():
    timer = RegionTimer(TIMING)
    assert feed(timer, 0, 4, BAD) == []
    assert feed(timer, 4.1, 20, SAFE) == []


def test_safe_posture_is_never_flagged():
    assert feed(RegionTimer(TIMING), 0, 120, SAFE) == []


def test_caution_is_flagged_only_after_a_minute():
    fired = feed(RegionTimer(TIMING), 0, 61, CAUTION)
    assert fired == [pytest.approx(60.0)]


def test_a_short_detection_gap_keeps_the_clock_running():
    timer = RegionTimer(TIMING)
    feed(timer, 0, 3, BAD)
    feed(timer, 3.1, 3.8, None)
    fired = feed(timer, 3.9, 6, BAD)
    assert timer.held(6.0) == pytest.approx(6.0)
    assert fired == [pytest.approx(5.0)]


def test_a_long_detection_gap_stops_the_clock():
    timer = RegionTimer(TIMING)
    feed(timer, 0, 3, BAD)
    feed(timer, 3.1, 5, None)
    assert timer.level is None
    fired = feed(timer, 5.1, 9, BAD)
    assert timer.held(9.0) == pytest.approx(3.9)
    assert fired == []


def test_time_totals_per_level():
    timer = RegionTimer(TIMING)
    feed(timer, 0, 2, SAFE)
    feed(timer, 2.1, 5, BAD)
    assert sum(timer.totals) == pytest.approx(5.0)
    assert timer.totals[BAD] == pytest.approx(2.4, abs=0.15)
    assert timer.totals[CAUTION] == 0


def run_monitor(monitor, worker, levels, labels, start, end, step=0.1):
    fired = []
    for i in range(round((end - start) / step) + 1):
        fired += monitor.update(round(start + i * step, 3), worker, levels, labels, clock="12:00:00")
    return fired


LABELS = {"back": ("Bent forward 70°", 70.0), "neck": ("Neutral", 3.0),
          "shoulders": ("Arm above shoulder 110°", 110.0), "knees": ("Deep knee bend 80°", 80.0)}


def test_flag_carries_worker_region_and_pod_code():
    monitor = PostureMonitor(TIMING)
    levels = {"back": BAD, "neck": SAFE, "shoulders": SAFE, "knees": SAFE}
    fired = run_monitor(monitor, 7, levels, LABELS, 0, 6)
    assert len(fired) == 1
    flag = fired[0]
    assert (flag["worker"], flag["region"], flag["level"], flag["pulses"]) == (7, "back", "bad", 2)
    assert flag["held_s"] == pytest.approx(5.0)
    assert flag["angle"] == 70
    assert flag["message"] == "Worker 7: bent forward 70° for 5 s"
    assert list(monitor.flags) == fired
    assert monitor.view(7, 6.0)["back"] == {
        "level": "bad", "label": "Bent forward 70°", "angle": 70, "held_s": 6.0, "flagged": True}
    assert monitor.view(7, 6.0)["neck"] == {
        "level": "safe", "label": "Neutral", "angle": 3, "held_s": 6.0, "flagged": False}


def test_words_follow_the_timed_posture_not_a_blip():
    monitor = PostureMonitor(TIMING)
    bent = {"back": BAD, "neck": SAFE, "shoulders": SAFE, "knees": SAFE}
    upright = {**bent, "back": SAFE}
    run_monitor(monitor, 1, bent, LABELS, 0, 3)
    blip_labels = {**LABELS, "back": ("Upright", 2.0)}
    monitor.update(3.1, 1, upright, blip_labels)  # one frame of standing up: too short to count
    back = monitor.view(1, 3.1)["back"]
    assert (back["level"], back["label"]) == ("bad", "Bent forward 70°")
    run_monitor(monitor, 1, upright, blip_labels, 3.2, 5)  # now it lasts
    back = monitor.view(1, 5.0)["back"]
    assert (back["level"], back["label"]) == ("safe", "Upright")


def test_hidden_regions_say_so():
    monitor = PostureMonitor(TIMING)
    monitor.update(0.0, 1, {"back": SAFE, "neck": None, "shoulders": SAFE, "knees": None}, LABELS)
    assert monitor.view(1, 0.0)["knees"]["label"] == "Not visible"


def test_flags_are_numbered_in_order_across_runs():
    levels = {"back": BAD, "neck": SAFE, "shoulders": BAD, "knees": SAFE}
    first = run_monitor(PostureMonitor(TIMING), 1, levels, LABELS, 0, 6)
    assert [f["region"] for f in first] == ["back", "shoulders"]
    assert first[1]["seq"] == first[0]["seq"] + 1
    later = run_monitor(PostureMonitor(TIMING), 1, levels, LABELS, 0, 6)  # e.g. after Stop and Start
    assert later[0]["seq"] == first[1]["seq"] + 1


def test_pod_pulse_codes_by_region():
    monitor = PostureMonitor(TIMING)
    levels = {"back": SAFE, "neck": SAFE, "shoulders": BAD, "knees": BAD}
    fired = run_monitor(monitor, 1, levels, LABELS, 0, 6)
    assert {f["region"]: f["pulses"] for f in fired} == {"shoulders": 1, "knees": 3}


def test_workers_are_timed_separately():
    monitor = PostureMonitor(TIMING)
    bad = {"back": BAD, "neck": SAFE, "shoulders": SAFE, "knees": SAFE}
    run_monitor(monitor, 1, bad, LABELS, 0, 4)
    run_monitor(monitor, 2, bad, LABELS, 3, 4)
    assert monitor.view(1, 4.0)["back"]["held_s"] == pytest.approx(4.0)
    assert monitor.view(2, 4.0)["back"]["held_s"] == pytest.approx(1.0)


def test_static_posture_after_a_minute():
    monitor = PostureMonitor(TIMING)
    levels = {"back": CAUTION, "neck": SAFE, "shoulders": SAFE, "knees": SAFE}
    run_monitor(monitor, 1, levels, LABELS, 0, 59)
    assert not monitor.is_static(1, 59.0)
    run_monitor(monitor, 1, levels, LABELS, 59.1, 61)
    assert monitor.is_static(1, 61.0)
    assert not monitor.is_static(99, 61.0)


def test_workers_who_leave_are_forgotten():
    monitor = PostureMonitor(TIMING)
    levels = {"back": SAFE, "neck": SAFE, "shoulders": SAFE, "knees": SAFE}
    monitor.update(0.0, 1, levels, LABELS)
    monitor.update(2.0, 2, levels, LABELS)
    assert monitor.prune(4.0) == [1]
    assert list(monitor.workers) == [2]


def test_describe_names_each_region():
    cfg = AngleConfig()
    bent = Angles(view="side", trunk_flexion=68.0, trunk_side_bend=0.0, neck_flexion=28.0, neck_side_bend=0.0,
                  upper_arm={"l": 110.0, "r": 10.0}, knee={"l": 75.0, "r": 70.0})
    text = {region: label for region, (label, _) in describe(bent, cfg).items()}
    assert text == {"back": "Bent forward 68°", "neck": "Neck bent 28°",
                    "shoulders": "Arm above shoulder 110°", "knees": "Deep knee bend 70°"}

    neutral = Angles(view="side", trunk_flexion=2.0, trunk_side_bend=0.0, neck_flexion=4.0, neck_side_bend=0.0,
                     upper_arm={"l": 8.0, "r": 5.0}, knee={"l": 3.0, "r": 4.0})
    text = {region: label for region, (label, _) in describe(neutral, cfg).items()}
    assert text == {"back": "Upright", "neck": "Neutral", "shoulders": "Arms down", "knees": "Legs straight"}

    hidden = {region: label for region, (label, _) in describe(Angles(), cfg).items()}
    assert set(hidden.values()) == {"Not visible"}
