"""Finite-plan and validation-selection behavior without launching training."""
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'experiments/computers/run.py'
spec = importlib.util.spec_from_file_location('computers_entry', path)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_resume_rejects_changed_plan_without_overwriting(tmp_path):
    plan = {'learning_rates': [.01, .048], 'wall_seconds': 10800}
    entry.freeze_plan(tmp_path, plan)
    before = (tmp_path / 'plan.json').read_bytes()
    entry.freeze_plan(tmp_path, plan)
    with pytest.raises(ValueError, match='Plan changed'):
        entry.freeze_plan(tmp_path, {**plan, 'learning_rates': [.096]})
    assert (tmp_path / 'plan.json').read_bytes() == before


def test_selection_waits_for_complete_grid_and_preserves_ties(tmp_path):
    plan = {'learning_rates': [.01, .048]}
    for lr in plan['learning_rates']:
        run = tmp_path / 'trials' / f'lr{lr:g}'
        run.mkdir(parents=True)
        (run / 'metrics.json').write_text(json.dumps({'complete': lr == .01, 'best_val_accuracy': .9, 'best_epoch': 10}))
    assert entry.summarize(tmp_path, plan)['selected'] is None
    p = tmp_path / 'trials/lr0.048/metrics.json'
    p.write_text(json.dumps({'complete': True, 'best_val_accuracy': .9, 'best_epoch': 20}))
    result = entry.summarize(tmp_path, plan)
    assert result['complete'] and result['selected']['base_lr'] == .01
