import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.registry as registry

# Define a test register
registry.REGISTRY_PATH = "tests/test_registry.npz"

def test_register_and_search():
    fake_embedding = np.random.rand(128)
    registry.register("Steve", fake_embedding)
    result = registry.search(fake_embedding)
    assert result is not None
    assert result["name"] == "Steve"
    print("register and search: PASSED")


def test_no_match():
    random_embedding = np.ones(128) * 999
    result = registry.search(random_embedding)
    assert result is None
    print("no match beyond threshold: PASSED")

def test_register_multi():
    # Small base vector (model emits non-unit-norm embeddings) + per-sample noise
    base = np.random.rand(128) * 0.2
    samples = [base + np.random.normal(0, 0.01, 128) for _ in range(10)]
    registry.register_multi("Steve", samples)

    # Stored template is the raw mean, in the same scale as the samples
    mean = np.stack(samples).mean(axis=0)
    result = registry.search(mean)
    assert result is not None
    assert result["name"] == "Steve"

    # A single noisy sample should still match the averaged template
    result = registry.search(samples[0])
    assert result is not None
    assert result["name"] == "Steve"

    index = registry.names.index("Steve")
    assert np.allclose(registry.embeddings[index], mean)

    registry.delete("Steve")
    print("register_multi (raw averaged template): PASSED")

def test_delete():
    registry.delete("Steve")
    result = registry.search(np.random.rand(128))
    assert result is None
    assert not os.path.exists(registry.REGISTRY_PATH)
    print("delete: PASSED")

test_register_and_search()
test_no_match()
test_register_multi()
test_delete()