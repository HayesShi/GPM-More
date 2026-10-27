import numpy as np
import pytest

from gpm_more.data import PreparedData, prepare, selected_labels


def test_only_allowed_label_rows_are_decoded(tmp_path):
    graph = tmp_path/'graph.npz'
    np.savez_compressed(graph, x=np.eye(5, dtype=np.float32),
                        edge_index=np.array([[0,1,2],[1,2,0]]),
                        train_mask=np.array([1,0,1,0,0], dtype=bool),
                        val_mask=np.array([0,1,0,0,0], dtype=bool),
                        y=np.array([1.,2.,3.,np.nan,np.nan]))
    np.testing.assert_array_equal(selected_labels(graph, np.array([2,0]), 5), [3.,1.])
    bank = tmp_path/'bank'
    bank.mkdir()
    for prefix, count in [('context',16),('extra',8)]:
        paths = np.broadcast_to(np.arange(5)[:,None,None], (5,count,9)).copy()
        relations = np.zeros((5,count,93), dtype=np.float16)
        relations[...,92] = 1
        np.save(bank/(prefix+'_paths.npy'), paths)
        np.save(bank/(prefix+'_relations.npy'), relations)
    folder = tmp_path/'prepared'
    prepare(graph, folder, name='label_boundary', bank=bank)
    data = PreparedData(folder)
    np.testing.assert_array_equal(data.targets([2,1,0]), [3.,2.,1.])
    with pytest.raises(ValueError, match='train/validation'):
        data.targets([3])
    paths, relations, actions = data.batch([0,2], 16, np.array([0,1]))
    assert paths.shape == (2,16,9) and relations.shape == (2,16,93)
    np.testing.assert_array_equal(actions[1,8:], 1)
    np.testing.assert_array_equal(actions[0], 0)


def test_fractional_validation_class_is_rejected_before_cache_creation(tmp_path):
    graph = tmp_path/'fractional_class.npz'
    np.savez(graph, x=np.eye(4, dtype=np.float32),
             edge_index=np.array([[0,1,2],[1,2,3]]),
             train_ids=np.array([0,1]), val_ids=np.array([2]),
             train_y=np.array([0,1]), val_y=np.array([.5]))
    destination = tmp_path/'prepared'
    with pytest.raises(ValueError, match='nonnegative integers'):
        prepare(graph, destination, task='multiclass_classification')
    assert not destination.exists()
