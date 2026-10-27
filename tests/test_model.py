import torch

from gpm_more import ModelConfig, RelationalPatternModel, closure_phase_pool, fixed_frequencies


def brute_force(paths, contextual, frequencies):
    batch, count, _, hidden = contextual.shape
    result = contextual.new_zeros(batch, hidden)
    if count == 1:
        return result
    for b in range(batch):
        for p in range(count):
            for q in range(count):
                if p == q:
                    continue
                for t in range(1, 9):
                    for s in range(1, 9):
                        if paths[b, p, t] == paths[b, q, s]:
                            result[b] += (contextual[b, p, t] * contextual[b, q, s]
                                          * ((t+s) * frequencies).cos())
    return result / (64 * count * (count-1))


def test_closure_matches_literal_distinct_walk_pairs_and_singleton():
    torch.manual_seed(1)
    # Repeated visits, two identical walks, and root revisits are intentional.
    paths = torch.tensor([[[0, 1, 2, 1, 0, 3, 2, 1, 0],
                           [0, 1, 2, 1, 0, 3, 2, 1, 0],
                           [0, 4, 5, 4, 0, 6, 7, 1, 0]],
                          [[8, 8, 9, 9, 8, 9, 8, 9, 8],
                           [8, 9, 8, 9, 8, 9, 8, 9, 8],
                           [8, 8, 9, 9, 8, 9, 8, 9, 8]]])
    contextual = torch.randn(2, 3, 9, 7, dtype=torch.float64, requires_grad=True)
    frequency = fixed_frequencies(7, dtype=torch.float64)
    pooled = closure_phase_pool(paths, contextual, frequency)
    torch.testing.assert_close(pooled, brute_force(paths, contextual, frequency), rtol=1e-12, atol=1e-12)
    singleton = closure_phase_pool(paths[:, :1], contextual[:, :1], frequency)
    assert torch.count_nonzero(singleton) == 0
    singleton.sum().backward()
    assert torch.count_nonzero(contextual.grad) == 0


def fixture_model():
    torch.manual_seed(21)
    model = RelationalPatternModel(ModelConfig(input_dim=5, num_outputs=2,
                                               hidden_dim=16, pattern_dropout=0.0))
    x = torch.randn(14, 5)
    paths = torch.randint(0, 14, (2, 4, 9))
    paths[:, :, 0] = torch.tensor([0, 1])[:, None]
    relations = torch.randn(2, 4, 93)
    return model, x, paths, relations


def test_model_forward_is_identical_while_feedback_gradient_changes():
    model, x, paths, relations = fixture_model()
    model.eval()
    outputs, grads = {}, {}
    for alpha in (0., .5, 1.):
        model.zero_grad(set_to_none=True)
        output = model(x, paths, relations, feedback_alpha=alpha)
        output['prediction'].square().sum().backward()
        outputs[alpha] = output['prediction'].detach().clone()
        grads[alpha] = {name: value.grad.detach().clone()
                        for name, value in model.named_parameters() if value.grad is not None}
    assert torch.equal(outputs[0.], outputs[1.])
    assert torch.equal(outputs[.5], outputs[1.])
    name = 'backbone.pattern_encoder.pre_projection.weight'
    assert not torch.allclose(grads[0.][name], grads[1.][name])
    torch.testing.assert_close(grads[.5][name], (grads[0.][name] + grads[1.][name]) / 2,
                               rtol=3e-5, atol=3e-7)
    # Ordinary-only identity PE, observed relation adapter and head still learn.
    for name in ['backbone.pattern_encoder.pe_projection.weight',
                 'relation_adapter.0.weight', 'backbone.head.weight']:
        assert grads[0.][name].norm() > 0
        torch.testing.assert_close(grads[0.][name], grads[1.][name], rtol=0, atol=0)


def test_node_relabeling_preserves_predictions():
    model, x, paths, relations = fixture_model()
    model.eval()
    permutation = torch.tensor([7, 12, 3, 0, 11, 6, 8, 13, 10, 4, 9, 2, 1, 5])
    permuted_x = torch.empty_like(x)
    permuted_x[permutation] = x
    with torch.no_grad():
        original = model(x, paths, relations)['prediction']
        relabeled = model(permuted_x, permutation[paths], relations)['prediction']
    torch.testing.assert_close(relabeled, original, rtol=2e-6, atol=2e-7)


def test_historical_loader_ignores_only_known_inactive_keys():
    model, _, _, _ = fixture_model()
    state = dict(model.state_dict())
    state['backbone.encoder_layer.unused'] = torch.tensor(0.)
    assert model.load_historical_state(state) == ['backbone.encoder_layer.unused']
