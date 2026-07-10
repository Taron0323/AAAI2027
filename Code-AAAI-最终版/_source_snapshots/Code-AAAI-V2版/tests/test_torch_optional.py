from foreact.models.foreact_torch import TorchUnavailable, require_torch


def test_torch_optional_import_contract():
    try:
        torch, nn = require_torch()
    except TorchUnavailable as exc:
        assert "[torch]" in str(exc)
    else:
        assert hasattr(torch, "softmax")
        assert hasattr(nn, "Module")
