import numpy as np

from halter.perception.pointcloud import backproject, fuse_object_points
from halter.scene_graph.relations import RelationConfig, infer_relations
from halter.scene_graph.serialize import graph_diff, graph_to_text
from halter.types import ObjectNode, RGBDFrame, SceneGraph


def box(name, lo, hi, n=400, seed=0):
    rng = np.random.default_rng(seed)
    node = ObjectNode(name=name)
    node.update_geometry(rng.uniform(lo, hi, size=(n, 3)))
    return node


def test_on_in_under():
    cfg = RelationConfig(containers=["pot"], supports=["plate", "stove"], table_z=0.0)
    nodes = {
        "plate": box("plate", [0.4, 0.0, 0.0], [0.6, 0.2, 0.02]),
        "lemon": box("lemon", [0.47, 0.07, 0.02], [0.53, 0.13, 0.08]),
        "pot": box("pot", [0.0, 0.0, 0.0], [0.2, 0.2, 0.15]),
        "carrot": box("carrot", [0.05, 0.05, 0.01], [0.15, 0.08, 0.05]),
        "cup": box("cup", [0.8, 0.0, 0.0], [0.85, 0.05, 0.1]),
    }
    t = {r.as_tuple() for r in infer_relations(nodes, cfg)}
    assert ("lemon", "on", "plate") in t
    assert ("carrot", "in", "pot") in t
    assert ("cup", "on", "table") in t
    assert ("carrot", "on", "table") not in t


def test_drawer_state_and_region():
    cfg = RelationConfig(containers=["drawer"], table_z=None,
                         container_regions={"drawer": {"min": [0.5, -0.1, 0.0], "max": [0.8, 0.1, 0.1]}},
                         articulated={"drawer": {"axis": [-1, 0, 0], "closed_pos": -0.5, "open_threshold": 0.05}})
    closed = {"drawer": box("drawer", [0.48, -0.1, 0.0], [0.52, 0.1, 0.1]),
              "berry": box("berry", [0.6, -0.02, 0.01], [0.64, 0.02, 0.04])}
    t = {r.as_tuple() for r in infer_relations(closed, cfg)}
    assert closed["drawer"].attributes["state"] == "closed"
    assert ("berry", "in", "drawer") in t
    opened = {"drawer": box("drawer", [0.33, -0.1, 0.0], [0.37, 0.1, 0.1]),
              "berry": box("berry", [0.45, -0.02, 0.01], [0.49, 0.02, 0.04])}
    t = {r.as_tuple() for r in infer_relations(opened, cfg)}
    assert opened["drawer"].attributes["state"] == "open"
    assert ("berry", "in", "drawer") in t   # region moved with the drawer


def test_backproject_and_fuse():
    K = np.array([[500, 0, 32], [0, 500, 24], [0, 0, 1.0]])
    depth = np.full((48, 64), 1.0, np.float32)
    mask = np.zeros((48, 64), bool)
    mask[10:30, 10:40] = True
    pts = backproject(depth, mask, K, np.eye(4))
    assert np.allclose(pts[:, 2], 1.0) and len(pts) == mask.sum()
    f = RGBDFrame("c", np.zeros((48, 64, 3), np.uint8), depth, K, np.eye(4))
    out = fuse_object_points([f, f], [{"a": mask}, {"a": mask}], voxel=0.001, min_points=10)
    assert "a" in out and len(out["a"]) > 10


def test_serialize():
    g = SceneGraph(nodes={"lemon": box("lemon", [0, 0, 0], [0.1, 0.1, 0.1])})
    from halter.types import Relation
    g.relations = [Relation("lemon", "on", "plate")]
    txt = graph_to_text(g)
    assert "(lemon, on, plate)" in txt and "centroid_m" in txt
    g2 = SceneGraph(nodes=g.nodes, relations=[Relation("lemon", "in", "drawer")])
    d = graph_diff(g, g2)
    assert "(lemon, in, drawer)" in d and "(lemon, on, plate)" in d
