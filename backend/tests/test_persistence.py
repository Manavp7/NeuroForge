from neuroforge.data.synthetic import SyntheticPatientGenerator
from neuroforge.models import LoopRun
from neuroforge.persistence import Database


def test_patient_roundtrip_excludes_latent():
    db = Database(":memory:")
    profile = SyntheticPatientGenerator(seed=3).generate("parkinsonian")
    db.save_patient(profile)
    loaded = db.get_patient(profile.id)
    assert loaded is not None
    assert "latent_state" not in loaded
    assert loaded["condition"] == "parkinsonian"
    assert any(p["id"] == profile.id for p in db.list_patients())


def test_run_roundtrip():
    db = Database(":memory:")
    run = LoopRun(id="r1", patient_id="p1", status="stabilized")
    db.save_run(run)
    loaded = db.get_run("r1")
    assert loaded["status"] == "stabilized"
    assert db.list_runs()[0]["id"] == "r1"


def test_audit_hash_chain():
    db = Database(":memory:")
    db.append_audit("r1", "clinician", "approve", "cand-1", "ok")
    db.append_audit("r1", "clinician", "reject", "cand-2", "no")
    records = db.list_audit("r1")
    assert len(records) == 2
    assert db.verify_audit() is True
    # Tamper and confirm detection.
    db.conn.execute("UPDATE audit SET action='approve' WHERE candidate_id='cand-2'")
    db.conn.commit()
    assert db.verify_audit() is False


def test_file_persistence_survives_reopen(tmp_path):
    path = str(tmp_path / "nf.db")
    db1 = Database(path)
    run = LoopRun(id="r9", patient_id="p9", status="running")
    db1.save_run(run)
    db1.conn.close()
    db2 = Database(path)
    assert db2.get_run("r9") is not None
