from neuroforge.synthetic import CHANNEL_NAMES, SyntheticPatientGenerator


def test_seeded_profile_and_first_iteration_are_deterministic() -> None:
    generator = SyntheticPatientGenerator()
    profile_a = generator.generate_profile(seed=42)
    profile_b = generator.generate_profile(seed=42)

    assert profile_a == profile_b

    rng_a = generator.rng(42, step=0)
    rng_b = generator.rng(42, step=0)
    biomarkers_a = generator.generate_biomarkers(profile_a, step=0, rng=rng_a)
    biomarkers_b = generator.generate_biomarkers(profile_b, step=0, rng=rng_b)
    window_a = generator.generate_neural_window(profile_a, biomarkers_a, step=0, rng=rng_a)
    window_b = generator.generate_neural_window(profile_b, biomarkers_b, step=0, rng=rng_b)

    assert biomarkers_a == biomarkers_b
    assert window_a.band_powers == window_b.band_powers
    assert window_a.signals[0][:5] == window_b.signals[0][:5]


def test_generated_signal_window_shape_and_ranges() -> None:
    generator = SyntheticPatientGenerator(sample_rate_hz=64, window_seconds=1.5)
    profile = generator.generate_profile(seed=5)
    rng = generator.rng(5, step=2)
    biomarkers = generator.generate_biomarkers(profile, step=2, rng=rng)
    window = generator.generate_neural_window(profile, biomarkers, step=2, rng=rng)

    assert window.channel_names == CHANNEL_NAMES
    assert len(window.timestamps) == 96
    assert len(window.signals) == len(CHANNEL_NAMES)
    assert all(len(channel) == 96 for channel in window.signals)
    assert set(window.band_powers) == {"theta", "alpha", "beta", "gamma"}
    assert all(0 <= value <= 1 for value in window.band_powers.values())
    assert 0 <= biomarkers.inflammation <= 1
    assert 0 <= biomarkers.hrv <= 1
