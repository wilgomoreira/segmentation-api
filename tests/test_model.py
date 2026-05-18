import torch
import pytest
from src.model import UNet


@pytest.fixture
def model():
    """Create a UNet instance for testing."""
    return UNet(in_channels=3, out_channels=1)


def test_output_shape_single_image(model):
    """Output shape must match input spatial dimensions."""
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 1, 256, 256), f"Expected (1,1,256,256), got {out.shape}"


def test_output_shape_batch(model):
    """Model must handle any batch size correctly."""
    x = torch.randn(4, 3, 256, 256)
    out = model(x)
    assert out.shape == (4, 1, 256, 256), f"Expected (4,1,256,256), got {out.shape}"


def test_eval_mode(model):
    """Model must run correctly in eval mode as used by the API."""
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 256)


def test_output_is_logits(model):
    """Model must return raw logits, not probabilities."""
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    # Raw logits can be outside [0, 1]. If sigmoid was applied
    # inside the model all values would be between 0 and 1.
    has_values_outside_unit_range = (out > 1).any() or (out < 0).any()
    assert has_values_outside_unit_range, "Output looks like probabilities, expected raw logits"