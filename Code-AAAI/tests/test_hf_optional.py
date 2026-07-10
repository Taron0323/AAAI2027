from foreact.models.hf_backbone import HFUnavailable, require_transformers


def test_hf_optional_import_contract():
    try:
        require_transformers()
    except HFUnavailable as exc:
        assert "transformers" in str(exc)
