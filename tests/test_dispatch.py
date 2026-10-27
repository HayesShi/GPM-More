import json
import time

import pytest

from gpm_more.autotune import LockBusyError
from gpm_more.dispatch import prerequisite_status, run_dispatch


def test_dispatch_waits_for_predecessor_and_retries_busy_gpu_without_duplicate_runs(tmp_path):
    prior = tmp_path / 'prior'
    prior.mkdir()
    (prior / 'state.json').write_text(json.dumps({'status': 'running', 'jobs': {}}), encoding='utf8')
    recipe = {'schema': 'gpm-more-autotune-v1', 'datasets': [{'name': 'example', 'data': 'prepared',
        'init_weights': {str(seed): f'pretrain{seed}.pt' for seed in range(3)}}],
        'base_config': {'steps': 1000}, 'learning_rates': [3e-4], 'late_lr_factors': [1.],
        'stop_after': 400, 'seeds': [0, 1, 2], 'wall_hours': 1, 'cpu_threads': 1, 'device': 'cpu'}
    (tmp_path / 'recipe.json').write_text(json.dumps(recipe), encoding='utf8')
    plan = {'schema': 'gpm-more-dispatch-v1', 'prerequisites': ['prior'],
        'campaigns': [{'plan': 'recipe.json', 'output': 'campaign'}], 'poll_seconds': .001, 'wall_hours': 1}
    path = tmp_path / 'dispatch.json'
    path.write_text(json.dumps(plan), encoding='utf8')
    calls = []
    def sleep(_):
        (prior / 'state.json').write_text(json.dumps({'status': 'complete', 'jobs': {}}), encoding='utf8')
        time.sleep(.001)
    def queue(plan, output, **kwargs):
        calls.append(output)
        if len(calls) == 1:
            raise LockBusyError('GPU still held')
        assert (kwargs['_package_source'] / 'train.py').exists()
        return {'status': 'complete', 'complete_jobs': 6, 'failed_jobs': 0}
    out = tmp_path / 'dispatch'
    result = run_dispatch(path, out, _run_queue=queue, _sleep=sleep)
    assert result['status'] == 'complete' and len(calls) == 2
    assert result['planned_search_jobs'] == 2 and result['planned_confirmation_jobs'] == 4
    events = [json.loads(line)['status'] for line in (out / 'events.jsonl').read_text().splitlines()]
    assert 'waiting_prerequisite' in events and 'waiting_gpu_lock' in events
    run_dispatch(path, out, _run_queue=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('duplicate')), _sleep=sleep)
    recipe['learning_rates'] = [1e-4]
    (tmp_path / 'recipe.json').write_text(json.dumps(recipe), encoding='utf8')
    with pytest.raises(ValueError, match='frozen copy'):
        run_dispatch(path, out, _run_queue=queue, _sleep=sleep)


def test_prerequisite_holds_stop_low_disk_and_crash_but_allows_terminal_failure(tmp_path, monkeypatch):
    monkeypatch.setattr('gpm_more.dispatch._process_alive', lambda _: False)
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'status': 'running', 'pid': 123, 'jobs': {}}), encoding='utf8')
    assert prerequisite_status(tmp_path)['reason'] == 'waiting_for_queue_recovery'
    state.write_text(json.dumps({'status': 'failed', 'jobs': {}}), encoding='utf8')
    assert prerequisite_status(tmp_path)['ready']
    stop = tmp_path / 'STOP_REQUESTED'
    stop.touch()
    assert not prerequisite_status(tmp_path)['ready']
    stop.unlink()
    state.write_text(json.dumps({'status': 'low_disk', 'jobs': {}}), encoding='utf8')
    assert not prerequisite_status(tmp_path)['ready']
