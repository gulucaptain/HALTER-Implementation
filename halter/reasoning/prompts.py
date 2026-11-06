"""Prompt templates for the three LLM roles (same model, different prompts).

The paper does not release its prompts; these are written from the paper's description:
 - scoring:      completed-subtask fraction of the rollout, from the post-rollout scene graph
 - planning:     sequence atomic reset skills to bring the scene back to the canonical initial state
 - verification: (a) per-skill: did the expected effect appear in the new graph?
                 (b) final: does the final graph match the target initial state?
Each prompt is conditioned on a few reference cases (in-context examples).
"""

SYSTEM = (
    "You are the reasoning module of an autonomous robot evaluation harness. "
    "You receive a 3D scene graph of a tabletop scene, estimated from RGB-D cameras: object records "
    "(name, visibility, centroid and size in metres, articulation state) and spatial relation triples "
    "(subject, predicate, object) with predicates on / in / under. Objects marked visible=false are "
    "occluded and keep their last observation. Reason only from the graph. "
    "Always answer with a single JSON object and nothing else."
)

SCORING = """# Task: score a policy rollout
Task name: {task_name}
Task instruction given to the policy: "{instruction}"

The task consists of these ordered subtasks. A subtask counts as completed if its goal conditions hold
in the scene graph (or were necessarily achieved, e.g. a later subtask could not hold otherwise).
{subtasks}

## Reference cases
{references}

## Current post-rollout scene graph
{graph}

Return JSON: {{"completed_subtasks": [<subtask names>], "success": <true if all subtasks completed>,
"rationale": "<one or two sentences>"}}"""

PLANNING = """# Task: plan a scene reset
After a rollout of task "{task_name}", the scene must be reset to the canonical initial state below
using ONLY skills from the atomic reset skill library. Each skill is executed by a learned visuomotor
policy and only works when its preconditions hold. Skills are executed in order. Use the fewest skills.
If the scene is already in the initial state, return an empty plan. If the reset is impossible with the
library, return an empty plan and set "feasible": false.

## Target initial state (relations and attributes that must hold)
{target}

## Atomic reset skill library
{skills}

## Reference cases
{references}

## Current scene graph
{graph}

Return JSON: {{"plan": [<skill names in order>], "feasible": <true|false>,
"rationale": "<brief reasoning about which relations differ and how each skill fixes them>"}}"""

VERIFY_SKILL = """# Task: verify one reset skill execution
The robot just executed the skill "{skill_name}" ("{instruction}").
Expected effect: {expected}

## Reference cases of successful executions of this skill
{references}

## Scene graph BEFORE the skill
{before}

## Scene graph AFTER the skill
{after}

## Change summary (computed)
{diff}

Did the expected effect appear in the AFTER graph, without breaking other relations it should not touch?
Return JSON: {{"success": <true|false>, "rationale": "<brief>"}}"""

VERIFY_FINAL = """# Task: verify that the scene has been reset
## Target initial state
{target}

## Reference cases
{references}

## Current scene graph
{graph}

Does the current scene satisfy every target relation and attribute? Small position differences are fine.
Return JSON: {{"success": <true|false>, "missing": [<target conditions that do not hold>],
"rationale": "<brief>"}}"""
