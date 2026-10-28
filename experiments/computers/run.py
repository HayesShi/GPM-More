"""Run the retained Computers recipe, or a bounded validation-only LR search."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = json.loads((REPO / 'configs/computers_best.json').read_text(encoding='utf8'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def freeze_plan(output, plan):
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'plan.json'
    if path.exists():
        if read(path) != plan:
            raise ValueError('Plan changed. Resume with the same arguments or choose a new output directory.')
    else:
        path.write_text(json.dumps(plan, indent=2), encoding='utf8')


def summarize(output, plan):
    trials = []
    for rate in plan['learning_rates']:
        path = output / 'trials' / f'lr{rate:g}' / 'metrics.json'
        result = read(path) if path.exists() else {}
        trials.append({'base_lr': rate, 'peak_lr': rate / 160,
                       'complete': result.get('complete', False),
                       'validation_accuracy': result.get('best_val_accuracy'),
                       'best_epoch': result.get('best_epoch'), 'metrics': str(path)})
    complete = all(x['complete'] for x in trials)
    winner = max(trials, key=lambda x: x['validation_accuracy']) if complete else None
    return {'complete': complete, 'seed': 0, 'trials': trials, 'selected': winner,
            'selection': 'validation accuracy; first declared LR wins ties; no test evaluation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'status'])
    parser.add_argument('--output', type=Path, default=REPO / 'runs/computers')
    parser.add_argument('--graph', type=Path, help='Original Computers NPZ with x, edge_index and y')
    parser.add_argument('--prepared', type=Path, help='Reuse the complete historical shared graph/walk cache')
    parser.add_argument('--official-repo', type=Path, default=REPO / 'external/GPM')
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[DEFAULTS['base_lr']], help='Base LR; actual peak is base/160')
    parser.add_argument('--hours', type=float, default=DEFAULTS['wall_hours'])
    parser.add_argument('--gpu-lock', type=Path, default=REPO / 'runs/.gpu0.lock')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == 'status':
        print(json.dumps(read(output / 'summary.json') if (output / 'summary.json').exists()
                         else {'status': 'not_started'}, indent=2))
        return
    if args.graph is None or not args.graph.is_file():
        parser.error('--graph must name an existing Computers NPZ')
    if not 0 < args.hours <= 3 or not 1 <= len(args.learning_rates) <= 6:
        parser.error('Use at most six declared learning rates and a budget of at most three hours')
    if any(not 0 < x < 1 for x in args.learning_rates) or len(set(args.learning_rates)) != len(args.learning_rates):
        parser.error('Learning rates must be distinct positive values below1')
    official_repo = args.official_repo.resolve()
    revision = subprocess.check_output(['git', '-C', str(official_repo), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '2ed3ebf7df8d8401520e6ec3bdbd1c2b291b5784':
        raise ValueError('Fetch the pinned GPM revision with scripts/fetch_experiment_sources.py gpm')
    prepared = args.prepared.resolve() if args.prepared else output / 'prepared'
    if args.prepared and not (prepared / 'patterns_complete.json').exists():
        raise ValueError('--prepared must contain a complete graph and official128-SRW bank')
    plan = {'dataset': 'computers', 'split': 'official high column0', 'seed': 0,
            'initialization': 'fresh', 'feedback': 'full', 'learning_rates': args.learning_rates,
            'graph': str(args.graph.resolve()), 'prepared': str(prepared),
            'official_repo': str(official_repo), 'revision': revision,
            'wall_seconds': args.hours * 3600, 'task_head_seed': 70000000,
            'gpu_lock': str(args.gpu_lock.resolve())}
    # QueueLock protects the preparation, plan and all sequential trials.
    sys.path.insert(0, str(REPO / 'src'))
    from gpm_more.autotune import QueueLock
    output.mkdir(parents=True, exist_ok=True)
    with QueueLock(output / '.queue.lock'), QueueLock(args.gpu_lock):
        freeze_plan(output, plan)
        previous = summarize(output, plan)
        if previous['complete']:
            print(json.dumps(previous, indent=2)); return
        if (output / 'STOP_REQUESTED').exists():
            print('STOP_REQUESTED exists; no work started.'); return
        snapshot = output / 'code_snapshot'
        if not snapshot.exists():
            snapshot.mkdir()
            shutil.copytree(REPO / 'src/gpm_more', snapshot / 'gpm_more',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            for name in ['runner.py', 'rawsum_model.py', 'run.py']:
                shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
        elif (snapshot / 'run.py').read_bytes() != Path(__file__).read_bytes():
            raise ValueError('The entry point changed since this plan was frozen.')
        sys.path.insert(0, str(snapshot))
        # Drop the coordinator package imported above so model imports use its snapshot.
        for name in list(sys.modules):
            if name == 'gpm_more' or name.startswith('gpm_more.'):
                del sys.modules[name]
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        import runner
        import rawsum_model
        torch.set_num_threads(4)
        runner.ROOT = output
        runner.WORKSPACE = REPO
        runner.OFFICIAL = official_repo / 'GPM'
        runner.PREPARED = prepared
        runner.RUN = output / 'model'
        runner.SOURCE_GRAPH = args.graph.resolve()
        runner.SAFE_FEATURES = args.graph.resolve()
        budget_path = output / 'budget.json'
        if not budget_path.exists():
            now = time.time()
            runner._save_json(budget_path, {'started_unix': now, 'deadline_unix': now + plan['wall_seconds']})
        deadline = read(budget_path)['deadline_unix']

        def stop_reason(_=None):
            if (output / 'STOP_REQUESTED').exists(): return 'STOP_REQUESTED'
            if time.time() >= deadline - 180: return 'wall_budget_checkpoint_reserve'
            if shutil.disk_usage(output).free < 2 * 2**30: return 'disk_below_2GiB'
            return None

        reason = stop_reason()
        if reason:
            print(reason); return
        if args.prepared:
            meta = read(prepared / 'graph_complete.json')
            if meta.get('dataset') != 'computers' or meta.get('split') != 'high' or meta.get('split_column') != 0:
                raise ValueError('Prepared cache is not official Computers high column0')
        else:
            runner.prepare_graph()
            runner.prepare_patterns()
        rawsum_model.prepare_relations()
        for rate in args.learning_rates:
            reason = stop_reason()
            if reason: break
            run = output / 'trials' / f'lr{rate:g}'
            runner.train_seed0({'train_microbatch': 256, 'eval_microbatch': 8},
                model_factory=lambda params, seed=0: rawsum_model.make_model(params, seed, initialization='fresh'),
                run_dir=run, params_override={'lr': rate}, checkpoint_control=stop_reason,
                provenance={'method': 'raw_sum', 'initialization': 'fresh', 'feedback': 'full',
                            'task_head_seed': 70000000, 'PE8_initialization': 'zero',
                            'pretraining': 'none', 'not_bitwise_paired_training': True})
            result = summarize(output, plan)
            runner._save_json(output / 'summary.json', result)
            if not read(run / 'metrics.json')['complete']: break
            torch.cuda.empty_cache()
        result = summarize(output, plan)
        result['stop_reason'] = reason
        runner._save_json(output / 'summary.json', result)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
