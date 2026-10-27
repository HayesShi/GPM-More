"""Critical queue behavior; fake child results are engineering fixtures only."""
import copy
import json
import os

import pytest

from gpm_more.autotune import (QueueLock, _process_alive, inspect_inputs, load_plan,
                              run_queue, validate_pretraining_source)


def plan_fixture(tmp_path):
    plan = {"schema": "gpm-more-autotune-v1", "datasets": [
        {"name": "example", "data": "prepared", "init_weights": {
            str(seed): f"pretrain{seed}.pt" for seed in range(3)}}],
        "base_config": {"steps": 1000}, "learning_rates": [1e-4, 3e-4],
        "late_lr_factors": [1.], "stop_after": 400, "seeds": [0, 1, 2],
        "wall_hours": 1, "cpu_threads": 1, "device": "cpu", "min_free_gb": 0}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf8")
    return path, plan


def fake_inventory(_):
    return [{"dataset": "example", "task": "regression"}]


def test_autotune_rejects_nondeterministic_execution(tmp_path):
    path, plan = plan_fixture(tmp_path)
    plan['base_config']['deterministic'] = False
    path.write_text(json.dumps(plan), encoding='utf8')
    with pytest.raises(ValueError, match='deterministic execution'):
        load_plan(path)


def test_paper_aligned_source_requires_fixed_final_pretraining_epoch(tmp_path):
    execution = {'method': 'incidence_rotation', 'hidden_dim': 256, 'pretrain_epochs': 100,
        'pretrain_lr': .0003, 'optimizer': 'AdamW', 'schedule': 'constant',
        'logical_batch_size': 256, 'early_stopping': False,
        'pretrain_roots': 'all_nodes_unlabeled_full_graph_each_epoch_shuffled_without_replacement'}
    source = {'schema': 'paper-aligned-natural-gpm-v1', 'dataset': 'photo', 'seed': 0,
        'method': 'incidence_rotation', 'epoch': 100, 'updates': 3000,
        'initialization': 'independent_from_scratch_no_task_checkpoint_warmstart',
        'test_used_for_selection': False, 'protocol': {'execution': execution},
        **{name: {} for name in ['optimizer', 'numpy_rng_state', 'torch_rng_state', 'cuda_rng_states', 'python_rng_state']}}
    completion = {'complete': True, 'dataset': 'photo', 'seed': 0, 'method': 'incidence_rotation',
        'pretrain_epochs_completed': 100, 'pretrain_updates': 3000, 'test_used_for_selection': False}
    result = validate_pretraining_source(source, completion, tmp_path / 'latest.pt', 'photo', 0, 7650)
    assert result['source_selection'] == 'fixed_final_epoch100'
    with pytest.raises(ValueError, match='fixed final-epoch100'):
        validate_pretraining_source({**source, 'epoch': 15}, completion, tmp_path / 'latest.pt', 'photo', 0, 7650)
    with pytest.raises(ValueError, match='fixed final-epoch100'):
        validate_pretraining_source(source, completion, tmp_path / 'best.pt', 'photo', 0, 7650)


def write_result(job, folder, *, complete=True):
    folder.mkdir(exist_ok=True)
    (folder / "config.json").write_text(json.dumps(job["config"]), encoding="utf8")
    result = {"complete": complete, "completed_update": 1000 if complete else 50,
              "primary_metric": "R2", "best_update": 700,
              "best_validation": {"R2": job["config"]["lr"] * 1000},
              "last_validation": {"R2": 0.1}, "train_seconds": 1., "validation_seconds": .1,
              "stop_reason": None if complete else "stop_file"}
    (folder / "metrics.json").write_text(json.dumps(result), encoding="utf8")
    (folder / "latest.pt").write_bytes(b"fixture")


def test_queue_freezes_before_confirmation_and_does_not_duplicate_completed_jobs(tmp_path):
    path, plan = plan_fixture(tmp_path)
    output, launched = tmp_path / "output", []
    def executor(job, folder, config, command):
        if job["phase"] == "confirmation":
            selection = json.loads((output / "selections/example.json").read_text())
            assert selection["confirmation_seeds"] == [1, 2]
            assert job["config"]["lr"] == 3e-4
            assert len([item for item in launched if item["phase"] == "search"]) == 4
        launched.append(copy.deepcopy(job))
        write_result(job, folder)
        return 0
    first = run_queue(path, output, _executor=executor, _inspect=fake_inventory)
    assert first["status"] == "complete" and first["complete_jobs"] == 8
    assert len(launched) == len({job["id"] for job in launched}) == 8
    selection = (output / "selections/example.json").read_bytes()
    def forbidden(*_):
        pytest.fail("Completed job was relaunched")
    again = run_queue(path, output, _executor=forbidden, _inspect=forbidden)
    assert again["complete_jobs"] == 8
    assert selection == (output / "selections/example.json").read_bytes()
    plan["learning_rates"] = [1e-4]
    path.write_text(json.dumps(plan), encoding="utf8")
    with pytest.raises(ValueError, match="frozen"):
        run_queue(path, output, _executor=forbidden, _inspect=forbidden)


def test_paused_trial_resumes_its_checkpoint_and_failure_cannot_win(tmp_path):
    path, _ = plan_fixture(tmp_path)
    output = tmp_path / "output"
    def pause(job, folder, *_):
        write_result(job, folder, complete=False)
        return 0
    first = run_queue(path, output, _executor=pause, _inspect=fake_inventory)
    assert first["complete_jobs"] == 0 and first["status"] == "stop_file"
    resumed = []
    def finish(job, folder, config, command):
        if not resumed:
            assert "--resume" in command and "--reset-head" not in command
        resumed.append(job["id"])
        if job["family"] == "late_stop" and job["config"]["lr"] == 3e-4:
            return 1
        write_result(job, folder)
        return 0
    result = run_queue(path, output, _executor=finish, _inspect=fake_inventory)
    assert result["status"] == "failed" and result["failed_jobs"] == 1
    assert not (output / "selections/example.json").exists()
    assert len(resumed) == 4  # No confirmation after an incomplete search grid.


def test_lock_and_lingering_child_prevent_overlap(tmp_path):
    lock = tmp_path / "lock"
    with QueueLock(lock):
        with pytest.raises(RuntimeError, match="holds"):
            with QueueLock(lock):
                pass
    assert _process_alive(os.getpid())
    path, _ = plan_fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "STOP_REQUESTED").touch()
    result = run_queue(path, output, _executor=lambda *_: pytest.fail("Sentinel ignored"), _inspect=fake_inventory)
    assert result["status"] == "stopped" and result["complete_jobs"] == 0
    state_path = output / "state.json"
    state = json.loads(state_path.read_text())
    state["jobs"]["unfinished"] = {"status": "running", "child_pid": os.getpid()}
    state_path.write_text(json.dumps(state), encoding="utf8")
    with pytest.raises(RuntimeError, match="still running"):
        run_queue(path, output, _inspect=fake_inventory)


def test_inventory_accepts_relocated_pretraining_receipt_and_reset_head_shape(tmp_path):
    import numpy as np
    import torch
    from gpm_more.model import ModelConfig, RelationalPatternModel
    path, plan = plan_fixture(tmp_path)
    data = tmp_path / 'prepared'
    data.mkdir()
    np.savez(data / 'data.npz', x=np.zeros((2, 5), dtype='float32'),
             train_ids=np.array([0]), val_ids=np.array([1]), train_y=np.array([0.]), val_y=np.array([1.]))
    (data / 'metadata.json').write_text(json.dumps({'schema': 'gpm-more-data-v1',
        'task': 'regression', 'num_outputs': 1}), encoding='utf8')
    for prefix, count in [('context', 16), ('extra', 8)]:
        np.save(data / f'{prefix}_paths.npy', np.zeros((2, count, 9), dtype='int32'))
        np.save(data / f'{prefix}_relations.npy', np.zeros((2, count, 93), dtype='float16'))
    plan['base_config']['model'] = {'hidden_dim': 16}
    for seed in range(3):
        source = tmp_path / 'relocated' / f'seed{seed}' / 'pretrained.pt'
        source.parent.mkdir(parents=True)
        model = RelationalPatternModel(ModelConfig(input_dim=5, num_outputs=3, hidden_dim=16))
        torch.save({'schema': 'natural-pattern-checkpoint-v1', 'phase': 'pretrain',
            'step': 900, 'dataset': 'example', 'seed': seed, 'family': 'incidence_rotation',
            'test_labels_read': False, 'state_dict': model.state_dict()}, source)
        receipt = {'complete': True, 'phase': 'pretrain', 'steps': 1000, 'best_step': 900,
            'dataset': 'example', 'seed': seed, 'family': 'incidence_rotation',
            'test_labels_read': False, 'checkpoint': f'Z:\\old-machine\\seed{seed}\\pretrained.pt'}
        (source.parent / 'complete.json').write_text(json.dumps(receipt), encoding='utf8')
        plan['datasets'][0]['init_weights'][str(seed)] = str(source)
    path.write_text(json.dumps(plan), encoding='utf8')
    inventory = inspect_inputs(load_plan(path))
    assert inventory[0]['sources'][0]['selected_step'] == 900
    assert inventory[0]['sources'][0]['completed_pretrain_steps'] == 1000
    assert inventory[0]['sources'][0]['completion_recorded_checkpoint'].startswith('Z:')
