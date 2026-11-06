import json
import os

from halter.factory import _p
from halter.llm.client import LLMConfig, make_llm
from halter.llm.parsing import extract_json
from halter.reasoning.reasoner import Reasoner, validate_plan
from halter.reasoning.reference_cases import ReferenceSet
from halter.reasoning.task_spec import load_scene, load_task
from halter.sim.symbolic import SymbolicWorld


def setup(task="drawer_store"):
    t = load_task(_p(f"configs/tasks/{task}.yaml"))
    s = load_scene(_p(f"configs/scenes/{t.scene}.yaml"))
    return t, s, ReferenceSet.load(t.reference_cases)


def test_yaml_on_is_string():
    t, s, _ = setup()
    assert s.target.relations[0] == ("lemon", "on", "plate")


def test_extract_json():
    assert extract_json('<think>x</think> ok ```json\n{"a": 1}\n```')["a"] == 1
    assert extract_json('blah {"plan": ["a", "b"], "x": {"y": 2}} trailing')["x"]["y"] == 2


def test_validate_plan():
    t, s, _ = setup()
    w = SymbolicWorld(s, seed=0)
    w.rollout(t, p_subtask=1.0)           # lemon in closed drawer
    g = w.ground_truth()
    ok, _ = validate_plan(["open_lower_drawer", "lemon_drawer_to_plate", "close_lower_drawer"], g, s)
    assert ok
    ok, why = validate_plan(["lemon_drawer_to_plate"], g, s)
    assert not ok and "preconditions" in why
    ok, why = validate_plan(["fly_away"], g, s)
    assert not ok


def test_mock_reasoner_all_tasks():
    for task in ["drawer_store", "drawer_retrieve", "pot_cook", "pan_cook",
                 "heldout_drawer_swap", "heldout_carrot_in_pot", "heldout_pan_lid"]:
        t, s, refs = setup(task)
        R = Reasoner(make_llm(LLMConfig(backend="mock")), s, t, refs)
        for seed in range(10):
            w = SymbolicWorld(s, seed=seed)
            w.rollout(t, p_subtask=0.6)
            g = w.observe()
            assert R.score(g).completed_subtasks == t.completed_prefix(g)
            plan, _ = R.plan_with_check(g)
            assert plan.valid, (task, plan.invalid_reason, sorted(g.triples()))


def test_prompts_render():
    t, s, refs = setup("pot_cook")
    captured = []

    def fake(msgs):
        captured.append(msgs[-1]["content"])
        return json.dumps({"completed_subtasks": [], "plan": [], "success": False, "missing": []})
    R = Reasoner(make_llm(LLMConfig(backend="mock"), mock_fn=fake), s, t, refs)
    g = SymbolicWorld(s).observe()
    R.score(g); R.plan(g); R.verify_final(g); R.verify_skill(s.skills["lid_back"], g, g)
    assert "Atomic reset skill library" in captured[1] and "lid_back" in captured[1]
    assert "Reference cases" in captured[0] and "Case 1" in captured[0]
    assert "BEFORE" in captured[3]
