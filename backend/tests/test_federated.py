from neuroforge.federated.fedavg import make_site_data, run_federated


def test_site_data_shapes():
    X, Y, names = make_site_data(seed=10, n=30)
    assert X.shape[0] == 30
    assert Y.shape[1] == 6
    assert len(names) == X.shape[1]


def test_federated_approaches_centralized():
    rep = run_federated(n_sites=4, per_site=50, rounds=20, seed=3)
    # Federated should converge close to centralized and improve over the first round.
    assert rep.federated_history[-1] < rep.federated_history[0]
    assert rep.federated_mse < 0.02
    assert abs(rep.federated_mse - rep.centralized_mse) < 0.01


def test_dp_noise_degrades_gracefully():
    clean = run_federated(n_sites=4, per_site=50, rounds=20, seed=3, dp_sigma=0.0)
    noisy = run_federated(n_sites=4, per_site=50, rounds=20, seed=3, dp_sigma=0.05)
    # DP noise should not crash and keeps MSE finite (usually >= clean).
    assert noisy.federated_mse >= 0.0
    assert noisy.dp_sigma == 0.05
    assert clean.federated_mse <= noisy.federated_mse + 0.05
