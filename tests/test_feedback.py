import pytest
import torch
from torch import nn

from gpm_more import FeedbackSchedule, scale_feedback


def test_forward_identity_and_input_gradient_scaling():
    value = torch.tensor([[-2.0, 0.0, 3.0]], requires_grad=True)
    branch = nn.Linear(3, 2, bias=False)
    branch.weight.data.copy_(torch.tensor([[1., 2., 3.], [-1., 4., 2.]]))
    gradients = {}
    for alpha in (0.0, 0.25, 1.0):
        branch.zero_grad(set_to_none=True)
        value.grad = None
        controlled = scale_feedback(value, alpha)
        assert torch.equal(controlled, value)
        branch(controlled).sum().backward()
        gradients[alpha] = value.grad.clone(), branch.weight.grad.clone()
    for alpha in (0.0, 0.25):
        torch.testing.assert_close(gradients[alpha][0], gradients[1.0][0] * alpha)
        # Feedback control must not suppress learnable relation parameters.
        torch.testing.assert_close(gradients[alpha][1], gradients[1.0][1], rtol=0, atol=0)


def test_schedule_boundary_and_validation():
    schedule = FeedbackSchedule(mode="late_stop", stop_after=400)
    assert schedule(400) == 1.0
    assert schedule(401) == 0.0
    assert FeedbackSchedule(mode="constant", alpha=.4)(1) == .4
    with pytest.raises(ValueError):
        scale_feedback(torch.ones(2), -0.1)
    with pytest.raises(ValueError):
        FeedbackSchedule()(0)
