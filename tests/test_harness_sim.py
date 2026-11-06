from halter.factory import build_sim
from halter.harness import metrics


def test_sim_campaign(tmp_path):
    cfg = {"llm": "configs/llm/mock.yaml", "sim": {"p_skill_success": 0.85, "p_subtask": 0.6},
           "harness": {"log_dir": str(tmp_path)}}
    eps = []
    for i, t in enumerate(["configs/tasks/drawer_store.yaml", "configs/tasks/pot_cook.yaml"]):
        h, _ = build_sim(cfg, t, seed=i)
        eps += [e.to_dict() for e in h.run(20)]
    m = metrics.summarize(eps)
    assert m["episodes"] == 40
    assert m["scoring_acc"]["mean"] == 100.0          # oracle reasoning
    assert m["verification_acc"]["mean"] == 100.0
    assert m["plan_validity"]["mean"] == 100.0
    assert 30 < m["reset_sr"]["mean"] < 100            # limited only by skill success
    assert m["interventions"] == sum(not e["reset_success"] for e in eps)


def test_perfect_skills_reset_everything(tmp_path):
    cfg = {"llm": "configs/llm/mock.yaml", "sim": {"p_skill_success": 1.0, "p_subtask": 0.5},
           "harness": {"log_dir": str(tmp_path)}}
    h, _ = build_sim(cfg, "configs/tasks/heldout_drawer_swap.yaml")
    assert all(e.reset_success for e in h.run(15))
