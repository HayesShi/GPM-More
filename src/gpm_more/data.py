"""Portable train/validation-only data and cached structural observations."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np


def selected_labels(path: Path, ids: np.ndarray, node_count: int):
    """Decode selected rows of numeric y.npy; skipped rows are never instantiated."""
    ids = np.asarray(ids, dtype=np.int64)
    with zipfile.ZipFile(path) as archive, archive.open('y.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError('Unsupported NPY label format')
        trailing = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if shape[0] != node_count or dtype.hasobject or (fortran and trailing != 1):
            raise ValueError('Labels must be numeric, row-contiguous and match the nodes')
        start = stream.tell()
        row_bytes = trailing * dtype.itemsize
        values = np.empty((len(ids), *shape[1:]), dtype=dtype)
        order = np.argsort(ids)
        sorted_ids = ids[order]
        cuts = np.r_[0, np.flatnonzero(np.diff(sorted_ids) != 1) + 1, len(ids)]
        for left, right in zip(cuts[:-1], cuts[1:]):
            if left == right:
                continue
            stream.seek(start + int(sorted_ids[left]) * row_bytes)
            raw = stream.read((right-left) * row_bytes)
            if len(raw) != (right-left) * row_bytes:
                raise IOError('Truncated labels')
            values[order[left:right]] = np.frombuffer(raw, dtype=dtype).reshape((right-left, *shape[1:]))
    return values


def prepare(graph_path, output, *, task='regression', name='custom', seed=0,
            bank=None, chunk_size=512, standardize_targets=False, split_index=0):
    """Prepare an already feature-preprocessed NPZ graph, reusing or sampling walks.

    Graph keys: x, edge_index, train_mask, val_mask, y. Alternatively provide
    train_ids,val_ids,train_y,val_y. No test labels or test mask are needed.
    GraphLand official feature transforms should be applied before this entry.
    """
    graph_path, output = Path(graph_path), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new prepared-data directory')
    if task not in {'regression', 'binary_classification', 'multiclass_classification'}:
        raise ValueError('Unsupported task')
    if not isinstance(split_index, int) or split_index < 0:
        raise ValueError('split_index must be a nonnegative integer')
    source_split = {'column': split_index, 'columns': 1}
    with np.load(graph_path, allow_pickle=False) as graph:
        x = np.asarray(graph['x'], dtype=np.float32)
        edge_index = np.asarray(graph['edge_index'])
        if {'train_ids', 'val_ids', 'train_y', 'val_y'}.issubset(graph.files):
            if split_index != 0:
                raise ValueError('Explicit IDs provide a single split')
            train_ids = np.asarray(graph['train_ids'], dtype=np.int64)
            val_ids = np.asarray(graph['val_ids'], dtype=np.int64)
            train_y, val_y = graph['train_y'].copy(), graph['val_y'].copy()
        else:
            train_mask, val_mask = graph['train_mask'], graph['val_mask']
            if train_mask.shape != val_mask.shape or train_mask.shape[0] != len(x):
                raise ValueError('Train/validation masks must match the graph')
            if train_mask.ndim == 2:
                if split_index >= train_mask.shape[1]:
                    raise ValueError('split_index exceeds the available mask columns')
                source_split['columns'] = train_mask.shape[1]
                train_mask, val_mask = train_mask[:, split_index], val_mask[:, split_index]
            elif train_mask.ndim != 1 or split_index != 0:
                raise ValueError('Expected one-dimensional masks or an explicit mask column')
            train_ids = np.flatnonzero(train_mask)
            val_ids = np.flatnonzero(val_mask)
            train_y = selected_labels(graph_path, train_ids, len(x))
            val_y = selected_labels(graph_path, val_ids, len(x))
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite preprocessed x[N,F]')
    joined = np.r_[train_ids, val_ids]
    if not len(train_ids) or not len(val_ids) or len(np.unique(joined)) != len(joined):
        raise ValueError('Train/validation must be nonempty, unique and disjoint')
    if joined.min() < 0 or joined.max() >= len(x):
        raise ValueError('Label node out of range')
    if len(train_y) != len(train_ids) or len(val_y) != len(val_ids):
        raise ValueError('Target rows must match the corresponding IDs')
    if not np.isfinite(train_y).all() or not np.isfinite(val_y).all():
        raise ValueError('Nonfinite allowed targets')
    transform = None
    if task == 'regression' and standardize_targets:
        mean, std = float(np.mean(train_y)), float(np.std(train_y))
        std = std if std > 0 else 1.
        train_y, val_y = (train_y-mean)/std, (val_y-mean)/std
        transform = {'mean': mean, 'scale': std, 'fit_split': 'train'}
    if task == 'multiclass_classification':
        if (not np.all(train_y == np.floor(train_y))
                or not np.all(val_y == np.floor(val_y))
                or np.min(train_y) < 0 or np.min(val_y) < 0):
            raise ValueError('Class IDs must be nonnegative integers')
        outputs = int(np.max(train_y))+1
        if np.min(val_y) < 0 or np.max(val_y) >= outputs:
            raise ValueError('Validation class absent from the training class range')
    else:
        outputs = 1
    if task == 'binary_classification' and not np.isin(np.r_[train_y.reshape(-1), val_y.reshape(-1)], [0, 1]).all():
        raise ValueError('Binary targets must be zero/one')
    output.mkdir(parents=True)
    np.savez_compressed(output/'data.npz', x=x, train_ids=train_ids, val_ids=val_ids,
                        train_y=train_y, val_y=val_y)
    if bank is not None:
        for key in ('context_paths', 'context_relations', 'extra_paths', 'extra_relations'):
            source = Path(bank)/(key+'.npy')
            array = np.load(source, mmap_mode='r', allow_pickle=False)
            count, width = (16, 9) if key == 'context_paths' else (16, 93) if key == 'context_relations' else (8, 9) if key == 'extra_paths' else (8, 93)
            if array.shape != (len(x), count, width):
                raise ValueError(f'Unexpected {key} shape: {array.shape}')
            shutil.copyfile(source, output/source.name)
        observation = {'mode': 'provided_cache', 'new_sampling': False}
    else:
        from .sampling import CSRGraph, sample_observations
        csr = CSRGraph.from_edge_index(edge_index, len(x))
        arrays = {}
        for key, shape, dtype in [('context_paths', (len(x),16,9), 'int32'),
                                  ('context_relations', (len(x),16,93), 'float16'),
                                  ('extra_paths', (len(x),8,9), 'int32'),
                                  ('extra_relations', (len(x),8,93), 'float16')]:
            arrays[key] = np.lib.format.open_memmap(output/(key+'.npy'), mode='w+', dtype=dtype, shape=shape)
        rng = np.random.default_rng(seed)
        for left in range(0, len(x), chunk_size):
            roots = np.arange(left, min(len(x), left+chunk_size), dtype=np.int64)
            for prefix, count, action in [('context', 16, 'srw'), ('extra', 8, 'nbrw')]:
                sampled = sample_observations(csr, roots, count, action, int(rng.integers(0, 2**31)))
                arrays[prefix+'_paths'][roots] = sampled.paths
                arrays[prefix+'_relations'][roots] = sampled.relations
            print(json.dumps({'prepared_roots': left+len(roots), 'total_roots': len(x)}), flush=True)
        for array in arrays.values():
            array.flush()
        observation = {'mode': 'new_cache', 'seed': seed, 'chunk_size': chunk_size,
                       'matching_historical_cache': False}
    metadata = {'schema': 'gpm-more-data-v1', 'name': name, 'task': task,
                'num_outputs': outputs, 'nodes': len(x), 'features': x.shape[1],
                'train_roots': len(train_ids), 'val_roots': len(val_ids),
                'labels': 'train_and_validation_only', 'test_labels_decoded': False,
                'target_standardization': transform, 'observations': observation}
    metadata['source_split'] = source_split
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    return metadata


class PreparedData:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.metadata = json.loads((self.folder/'metadata.json').read_text(encoding='utf8'))
        if self.metadata['schema'] != 'gpm-more-data-v1':
            raise ValueError('Unexpected prepared data schema')
        with np.load(self.folder/'data.npz', allow_pickle=False) as arrays:
            for key in ('x', 'train_ids', 'val_ids', 'train_y', 'val_y'):
                setattr(self, key, arrays[key].copy())
        self.task, self.num_outputs = self.metadata['task'], self.metadata['num_outputs']
        ids = np.r_[self.train_ids, self.val_ids]
        labels = np.concatenate([self.train_y, self.val_y])
        order = np.argsort(ids)
        self._ids, self._labels = ids[order], labels[order]
        for key in ('context_paths', 'context_relations', 'extra_paths', 'extra_relations'):
            setattr(self, key, np.load(self.folder/(key+'.npy'), mmap_mode='r', allow_pickle=False))

    def targets(self, nodes):
        nodes = np.asarray(nodes, dtype=np.int64)
        positions = np.searchsorted(self._ids, nodes)
        if np.any(positions >= len(self._ids)) or not np.array_equal(self._ids[positions], nodes):
            raise ValueError('Target access is limited to train/validation roots')
        return self._labels[positions].copy()

    def batch(self, nodes, budget, actions=None):
        nodes = np.asarray(nodes, dtype=np.int64)
        if not 1 <= budget <= 16:
            raise ValueError('Visible walk budget must be in [1,16]')
        paths = np.array(self.context_paths[nodes, :budget], copy=True)
        relations = np.array(self.context_relations[nodes, :budget], dtype=np.float32, copy=True)
        action_ids = np.zeros((len(nodes), budget), dtype=np.int64)
        if actions is not None:
            actions = np.asarray(actions, dtype=np.int64)
            if budget != 16 or actions.shape != (len(nodes),) or not np.isin(actions, [0,1]).all():
                raise ValueError('Actions require a 16-walk batch and one 0/1 action per root')
            changed = actions == 1
            paths[changed, 8:] = self.extra_paths[nodes[changed]]
            relations[changed, 8:] = self.extra_relations[nodes[changed]]
            action_ids[:, 8:] = actions[:, None]
        return paths, relations, action_ids
