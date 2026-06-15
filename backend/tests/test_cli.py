from neuroforge.cli import main


def test_cli_demo_runs(capsys):
    rc = main(["demo", "--condition", "neuroinflammatory", "--iters", "3", "--seed", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "closed-loop demo" in out
    assert "Final status" in out


def test_cli_json(capsys):
    rc = main(["demo", "--condition", "mood_disorder", "--iters", "2", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"disclaimer"' in out
