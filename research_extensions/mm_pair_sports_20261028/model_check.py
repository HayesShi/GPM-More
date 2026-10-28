"""One small real CPU critical-path check; does not train a model."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .pair_model import PairSportsModel
    from .walks import CSRGraph, sample_walks
except ImportError:
    from pair_model import PairSportsModel
    from walks import CSRGraph, sample_walks


def run_check() -> dict:
    torch.set_num_threads(2)
    started = time.perf_counter()
    torch.manual_seed(8301)
    edges = np.asarray([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4),
                        (4, 0), (1, 4), (4, 5), (5, 6)], dtype=np.int64).T
    excluded = np.asarray([(0, 1), (3, 4)], dtype=np.int64).T
    graph = CSRGraph.from_edge_index(edges, 8)
    obs = sample_walks(graph, np.arange(8), 16, 808, excluded_edges=excluded)
    physically_kept = edges[:, ~np.asarray([(u, v) in {(0, 1), (3, 4)}
                                          for u, v in edges.T])]
    physical = CSRGraph.from_edge_index(physically_kept, 8)
    reference = sample_walks(physical, np.arange(8), 16, 808)
    physical_paths_equal = bool(np.array_equal(obs.paths, reference.paths))
    physical_relations_equal = bool(np.array_equal(obs.relations, reference.relations))
    assert physical_paths_equal and physical_relations_equal
    removed = {(0, 1), (1, 0), (3, 4), (4, 3)}
    assert not any((int(u), int(v)) in removed
                   for u, v in zip(obs.paths[..., :-1].ravel(), obs.paths[..., 1:].ravel()))
    for i in range(9):
        for j in range(9):
            leaked = np.zeros(obs.paths.shape[:2], dtype=bool)
            for u, v in removed:
                leaked |= (obs.paths[..., i] == u) & (obs.paths[..., j] == v)
            assert not np.any(obs.adjacency[..., i, j] & leaked)
    assert obs.excluded_directed_edges == 4
    assert obs.relations.shape == (8, 16, 93)
    assert not obs.occurrence_valid[7, :, 1:].any()

    x = torch.randn(8, 1024)
    paths = torch.from_numpy(obs.paths).long()
    relations = torch.from_numpy(obs.relations)
    valid = torch.from_numpy(obs.occurrence_valid)
    model = PairSportsModel()
    same_initialization = copy.deepcopy(model)
    assert all(torch.equal(value, same_initialization.state_dict()[key])
               for key, value in model.state_dict().items())
    model.eval()
    src = torch.tensor([0, 0, 1, 2, 3, 4, 7])
    dst = torch.tensor([1, 4, 5, 6, 5, 6, 0])
    with torch.no_grad():
        encoded = model.encode(x, paths, relations, valid)
        base = model.score(encoded, src, dst, False)
        joint = model.score(encoded, src, dst, True)
        additive = model.joint_score(encoded, src, dst)
        assert torch.allclose(joint, base + model.joint_scale * additive)
        symmetry_error = float((additive - model.joint_score(encoded, dst, src)).abs().max())
        assert symmetry_error < 2e-6
        assert not encoded["valid"][7].any()
        assert additive[-1].item() == 0
        automatically_valid = model.encode(x, paths, relations)
        assert torch.equal(encoded["valid"], automatically_valid["valid"])
        assert torch.allclose(encoded["left"], automatically_valid["left"])
        assert torch.all(encoded["ids"][:, 1:] >= encoded["ids"][:, :-1])

        # Rename global IDs jointly while preserving each original observation.
        # Features and walk identities move together; no semantic change occurs.
        rename = torch.tensor([4, 6, 0, 7, 2, 5, 1, 3])
        renamed_x = torch.empty_like(x)
        renamed_x[rename] = x
        renamed_encoded = model.encode(renamed_x, rename[paths], relations, valid)
        renamed_base = model.score(renamed_encoded, src, dst, False)
        renamed_joint = model.score(renamed_encoded, src, dst, True)
        id_invariance_error = float((joint - renamed_joint).abs().max())
        base_invariance_error = float((base - renamed_base).abs().max())
        assert id_invariance_error < 3e-6
        assert base_invariance_error < 3e-6

        # A cached root bank may be scattered into repeated endpoint rows.
        scatter = torch.tensor([3, 0, 3, 4, 0])
        repeated = model.encode(x, paths[scatter], relations[scatter], valid[scatter])
        repeated_z_error = float((repeated["z"] - encoded["z"][scatter]).abs().max())
        assert repeated_z_error < 3e-6

        # Known nonzero identity-interaction case: independent relabeling of one
        # endpoint removes the only shared witness but leaves z untouched.
        handcrafted = {
            "z": torch.zeros(2, 256), "ids": torch.tensor([[2, 9], [2, 9]]),
            "valid": torch.tensor([[True, False], [True, False]]),
            "left": torch.zeros(2, 2, 16), "right": torch.zeros(2, 2, 16),
        }
        handcrafted["left"][:, 0, 0] = 1
        handcrafted["right"][:, 0, 0] = 1
        known_shared = float(model.joint_score(handcrafted, [0], [1]).item())
        handcrafted["ids"][1, 0] = 3
        broken_shared = float(model.joint_score(handcrafted, [0], [1]).item())
        assert known_shared == 1.0 and broken_shared == 0.0

    # One actual differentiable forward/backward through all model components.
    # This is a check, with no optimizer or checkpoint update.
    model.train()
    encoded = model.encode(x, paths, relations, valid)
    scores = model.score(encoded, src, dst, True)
    loss = F.binary_cross_entropy_with_logits(scores, torch.tensor([1., 1., 0., 0., 1., 0., 0.]))
    loss.backward()
    finite_gradients = all(p.grad is None or bool(torch.isfinite(p.grad).all())
                           for p in model.parameters())
    assert finite_gradients
    gradient_norms = {
        "U": float(model.U.grad.norm()), "V": float(model.V.grad.norm()),
        "joint_scale": float(model.joint_scale.grad.abs()),
        "pre_projection": float(model.node_model.backbone.pattern_encoder.pre_projection.weight.grad.norm()),
        "anonymous_gru": float(model.node_model.backbone.pattern_encoder.pe_rnn.weight_ih_l0.grad.norm()),
        "relation_adapter": float(model.node_model.relation_adapter[0].weight.grad.norm()),
        "action_embedding": float(model.node_model.action_embedding.weight.grad.norm()),
        "outer_transformer": float(model.node_model.backbone.encoder[0].self_attn.in_proj_weight.grad.norm()),
        "pair_head": float(model.pair_head[0].weight.grad.norm()),
    }
    assert all(value > 0 and np.isfinite(value) for value in gradient_norms.values())
    assert all(torch.equal(value, same_initialization.state_dict()[key])
               for key, value in model.state_dict().items())
    return {
        "status": "passed", "check": "small_real_cpu_critical_path_no_training",
        "torch_version": torch.__version__, "seconds": time.perf_counter() - started,
        "graph_nodes": 8, "undirected_edges": 9, "excluded_undirected_edges": 2,
        "num_walks": 16, "walk_steps": 8, "relation_dim": 93,
        "physical_edge_deletion_paths_equal": physical_paths_equal,
        "physical_edge_deletion_relations_equal": physical_relations_equal,
        "no_excluded_edge_in_transitions_or_descriptors": True,
        "isolated_root_no_valid_witness": True, "isolated_joint_score": float(additive[-1]),
        "ordinary_joint_initial_parameter_equality": True,
        "joint_additive_score_equation": True, "joint_symmetry_max_error": symmetry_error,
        "global_id_renaming_score_max_error": id_invariance_error,
        "global_id_renaming_base_max_error": base_invariance_error,
        "cached_root_scatter_z_max_error": repeated_z_error,
        "known_same_witness_contraction": known_shared,
        "independent_id_relabel_contraction": broken_shared,
        "backward_finite": finite_gradients, "backward_gradient_norms": gradient_norms,
        "loss": float(loss.detach()), "witness_table_shape": list(encoded["left"].shape),
        "joint_scale_initial": float(model.joint_scale.detach()),
        "witness_projection_scale": model.witness_projection_scale,
        "all_parameters_unchanged_after_backward": True,
        "full_gpu_check": "delegated_to_parent_runner_not_performed_here",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("engineering_check.json"))
    args = parser.parse_args()
    result = run_check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
