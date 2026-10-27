import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gpm_more.train import _replace_file, _save_json, initialize_weights, task_metrics, train
from gpm_more.model import ModelConfig, RelationalPatternModel


def test_atomic_write_retries_transient_permission_error_and_keeps_permanent_error(tmp_path, monkeypatch):
    original = Path.replace
    calls = []
    def flaky(path, destination):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError('temporary Windows sharing violation')
        return original(path, destination)
    monkeypatch.setattr(Path, 'replace', flaky)
    monkeypatch.setattr('gpm_more.train.time.sleep', lambda _: None)
    target = tmp_path / 'state.json'
    _save_json(target, {'status': 'running'})
    assert json.loads(target.read_text()) == {'status': 'running'}
    assert len(calls) == 2
    def blocked(*_):
        raise PermissionError('persistent access denial')
    monkeypatch.setattr(Path, 'replace', blocked)
    temporary = tmp_path / 'state.json.tmp'
    temporary.write_text('new', encoding='utf8')
    with pytest.raises(PermissionError, match='persistent'):
        _replace_file(temporary, target, attempts=2)
    assert json.loads(target.read_text()) == {'status': 'running'}


def test_explicit_head_reset_restores_encoder_and_preserves_training_rng(tmp_path):
    source = RelationalPatternModel(ModelConfig(input_dim=5, num_outputs=3, hidden_dim=16))
    path = tmp_path / 'pretrained.pt'
    torch.save({'state_dict': source.state_dict(), 'phase': 'pretrain', 'seed': 0}, path)
    target = RelationalPatternModel(ModelConfig(input_dim=5, num_outputs=1, hidden_dim=16))
    before = torch.get_rng_state().clone()
    initialization = initialize_weights(target, path, reset_head=True, seed=2)
    assert torch.equal(before, torch.get_rng_state())
    assert initialization['task_head_seed'] == 70000002
    assert initialization['optimizer_history_imported'] is False
    for name, value in source.state_dict().items():
        if not name.startswith('backbone.head.'):
            assert torch.equal(value, target.state_dict()[name])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(70000002)
        reference = torch.nn.Linear(16, 1)
    assert torch.equal(reference.weight, target.backbone.head.weight)
    assert torch.equal(reference.bias, target.backbone.head.bias)
    broken = source.state_dict()
    broken.pop('backbone.pattern_encoder.pre_projection.weight')
    torch.save({'state_dict': broken}, path)
    with pytest.raises(ValueError, match='lacks active weights'):
        initialize_weights(target, path, reset_head=True)


def test_regression_mse_name_does_not_assume_standardized_targets():
    metrics = task_metrics(np.array([100., 102.]), np.array([99., 104.]), 'regression')
    assert metrics['MSE'] == 2.5
    assert 'MSE_standardized' not in metrics


def prepared_fixture(folder):
    folder.mkdir()
    rng = np.random.default_rng(91)
    x = rng.normal(size=(12, 5)).astype(np.float32)
    ids = np.arange(12)
    np.savez(folder / 'data.npz', x=x, train_ids=ids[:8], val_ids=ids[8:],
             train_y=x[:8, 0] + x[:8, 1], val_y=x[8:, 0] + x[8:, 1])
    for prefix, count in [('context', 16), ('extra', 8)]:
        paths = rng.integers(0, 12, (12, count, 9), dtype=np.int32)
        paths[:, :, 0] = ids[:, None]
        np.save(folder / f'{prefix}_paths.npy', paths)
        np.save(folder / f'{prefix}_relations.npy', rng.normal(size=(12, count, 93)).astype(np.float32))
    metadata = {'schema': 'gpm-more-data-v1', 'name': 'resume_engineering_fixture',
                'task': 'regression', 'num_outputs': 1, 'nodes': 12, 'features': 5,
                'train_roots': 8, 'val_roots': 4, 'labels': 'train_and_validation_only',
                'test_labels_decoded': False, 'target_standardization': None,
                'observations': {'mode': 'engineering_fixture'}}
    (folder / 'metadata.json').write_text(json.dumps(metadata), encoding='utf8')


def assert_tree_equal(first, second):
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    else:
        assert first == second


def test_resume_replays_feedback_dropout_and_optimizer_history_exactly(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        data = tmp_path / 'data'
        prepared_fixture(data)
        config = {'steps': 6, 'seed': 11, 'warmup': 2, 'batch_size': 4,
                  'micro_batch_size': 2, 'train_budgets': [4, 16], 'eval_every': 2,
                  'feedback': {'mode': 'late_stop', 'stop_after': 2},
                  'late_lr': {'after': 2, 'factor': .2},
                  'model': {'hidden_dim': 16, 'pattern_dropout': .1}}
        whole, split = tmp_path / 'whole', tmp_path / 'split'
        train(data, whole, config)
        partial = train(data, split, config, max_updates=3)
        assert partial['complete'] is False
        assert partial['last_validation_update'] == 2
        train(data, split, config, resume=split / 'latest.pt')
        a = torch.load(whole / 'latest.pt', map_location='cpu', weights_only=True)
        b = torch.load(split / 'latest.pt', map_location='cpu', weights_only=True)
        for key in ('model_state', 'optimizer', 'scheduler', 'rng', 'history', 'local_steps',
                    'best_state', 'best_metrics', 'best_update'):
            assert_tree_equal(a[key], b[key])
        assert a['update'] == b['update'] == 6
        with pytest.raises(ValueError, match='configuration'):
            train(data, split, {**config, 'steps': 7}, resume=split / 'latest.pt')
    finally:
        torch.set_num_threads(previous_threads)
