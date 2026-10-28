"""Unchanged raw_sum forward; fixed source/fresh initialization controls."""
import time
from scipy.sparse import csr_matrix
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import runner as official

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from gpm_more.model import ModelConfig, RelationalPatternModel
from gpm_more.sampling import PAIR_I, PAIR_J, _descriptors


class OfficialProtocolRawSum(nn.Module):
    def __init__(self, params):
        super().__init__()
        config=ModelConfig(input_dim=775,num_outputs=10)
        self.model_config=config.to_dict()
        self.core=RelationalPatternModel(config)
        self.relations=np.load(official.PREPARED/'raw_sum_relations.npy',mmap_mode='r',allow_pickle=False)

    def forward(self, graph, nodes, params, **kwargs):
        node_ids=nodes.detach().cpu().numpy()
        all_paths=params['pattern_set']['pattern'][:,nodes,:]
        count,batch,_=all_paths.shape
        relation=np.asarray(self.relations[node_ids])
        if kwargs['mode']=='train':
            idx=torch.randint(0,count,(batch,params['num_patterns']))
            paths=torch.stack([all_paths[idx[i],i,:] for i in range(batch)],dim=0)
            relation=relation[np.arange(batch)[:,None],idx.numpy()]
        else:
            paths=all_paths.permute(1,0,2).contiguous()
        x=torch.cat((graph.x,graph.pe),dim=-1)
        pred=self.core(x,paths.to(x.device),torch.as_tensor(np.array(relation,copy=True),device=x.device,dtype=x.dtype),
                       action_ids=None,feedback_alpha=1.0)
        return pred['prediction'],pred['z'],pred['pattern_tokens'],0.0


def make_model(params,seed=0,*,initialization='source'):
    from utils.utils import seed_everything
    if initialization not in ('source','fresh'):
        raise ValueError('Only the two frozen initialization choices are allowed')
    seed_everything(seed)
    model=OfficialProtocolRawSum(params)
    with torch.no_grad():
        model.core.backbone.pattern_encoder.pre_projection.weight[:,767:].zero_()
    if initialization=='source':
        saved=torch.load(ROOT/'source_active_weights.pt',map_location='cpu',weights_only=True)
        state=dict(saved['state_dict'])
        key='backbone.pattern_encoder.pre_projection.weight'
        original=state[key]
        if original.shape!=(256,767):
            raise ValueError('Unexpected frozen source projection shape')
        expanded=original.new_zeros(256,775)
        expanded[:,:767]=original
        state[key]=expanded
        model.core.load_historical_state(state)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(70000000+seed)
        head=nn.Linear(256,10)
    model.core.backbone.head.load_state_dict(head.state_dict(),strict=True)
    return model.to(params['device'])


def prepare_relations():
    dest = official.PREPARED / "raw_sum_relations.npy"
    marker = official.PREPARED / "raw_sum_relations_complete.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf8"))
    paths = np.load(official.PREPARED / "paths.npy", mmap_mode="r", allow_pickle=False)
    with np.load(official.PREPARED / "shared_graph.npz", allow_pickle=False) as graph:
        edges = graph["edge_index"]
        n = len(graph["x"])
    matrix = csr_matrix((np.ones(edges.shape[1], dtype=np.bool_), (edges[0], edges[1])), shape=(n, n))
    matrix.sort_indices()
    result = np.lib.format.open_memmap(dest, mode="w+", dtype="float16", shape=(n, 128, 93))
    start = time.perf_counter()
    for left in range(0, n, 64):
        p = np.asarray(paths[left:left + 64])
        rows, cols = p[..., PAIR_I].reshape(-1), p[..., PAIR_J].reshape(-1)
        exists = np.asarray(matrix[rows, cols]).reshape(*p.shape[:2], 36)
        adjacency = np.zeros((*p.shape[:2], 9, 9), dtype=np.bool_)
        adjacency[..., PAIR_I, PAIR_J] = exists
        adjacency[..., PAIR_J, PAIR_I] = exists
        result[left:left + len(p)] = _descriptors(p, adjacency)[3].astype(np.float16)
    result.flush()
    metadata = {"complete": True, "shape": list(result.shape), "dtype": "float16",
                "same_official_paths": True, "equality_bits": 36, "induced_edge_bits": 36,
                "rooted_four_node_indicators": 21, "membership_probes": int(n * 128 * 36),
                "seconds": time.perf_counter() - start, "labels_loaded": False}
    official._save_json(marker, metadata)
    print(json.dumps({"stage": "raw_sum_relations_ready", **metadata}), flush=True)
    return metadata
