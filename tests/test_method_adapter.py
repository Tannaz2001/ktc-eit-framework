import numpy as np

from src.ktc_framework.adapters.method_adapter import MethodAdapter
from src.ktc_framework.registry import get_method, load_external_methods


class DictReturningMethod:
    def reconstruct(self, batch):
        return {"reconstruction": np.ones((256, 256), dtype=np.float32)}


class InvalidShapeMethod:
    def reconstruct(self, batch):
        return np.ones((16, 16), dtype=np.uint8)


def test_method_adapter_accepts_dict_reconstruction():
    reconstruction = MethodAdapter(DictReturningMethod()).reconstruct(batch=None)

    assert reconstruction.shape == (256, 256)
    assert reconstruction.dtype == np.uint8


def test_method_adapter_rejects_invalid_shape():
    try:
        MethodAdapter(InvalidShapeMethod()).reconstruct(batch=None)
    except ValueError as exc:
        assert "must return shape (256, 256)" in str(exc)
    else:
        raise AssertionError("Expected invalid reconstruction shape to fail")


def test_load_external_methods_registers_real_runtime_method():
    load_external_methods(["external_methods"])

    method_cls = get_method("DampedLeastSquaresReconstruction")

    assert method_cls.__name__ == "DampedLeastSquaresReconstruction"
    assert hasattr(method_cls(), "reconstruct")
