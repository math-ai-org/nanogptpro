import os

import numpy as np
import pytest

from lm_eval.api.instance import Instance
from lm_eval.caching import cache


def test_long_cache_keys_are_hashed_below_filesystem_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    long_model_name = "tokenizer" + "very-long-model-name/" * 30
    cache_key = f"requests-mmlu-5shot-rank0-world_size1-chat_template-{long_model_name}"

    cache.save_to_cache(cache_key, {"ok": True})

    cache_files = os.listdir(tmp_path)
    assert len(cache_files) == 1
    assert len(cache_files[0].encode("utf-8")) <= cache.MAX_CACHE_FILENAME_BYTES
    assert cache.load_from_cache(cache_key, cache=True) == {"ok": True}


def test_short_cache_keys_keep_readable_name(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    cache_key = "requests-sciq-0shot-rank0-world_size1-tokenizergpt2"
    cache.save_to_cache(cache_key, ["cached"])

    cache_files = os.listdir(tmp_path)
    assert cache_files == [f"{cache_key}{cache.FILE_SUFFIX}"]
    assert cache.load_from_cache(cache_key, cache=True) == ["cached"]


def test_request_cache_roundtrips_instances_as_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    instance = Instance(
        request_type="generate_until",
        doc={"question": "Q"},
        arguments=("context", {"until": ["stop"]}),
        idx=7,
        metadata=("task", 3, 1),
    )
    instance.resps = [["answer"]]
    instance.filtered_resps = {"exact_match": "answer"}

    cache.save_to_cache("requests-json", [[instance]])

    cache_files = os.listdir(tmp_path)
    assert cache_files == [f"requests-json{cache.FILE_SUFFIX}"]
    assert cache_files[0].endswith(".json")

    cached = cache.load_from_cache("requests-json", cache=True)
    assert cached == [[instance]]


def test_request_cache_serializes_numpy_torch_and_bytes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    instance = Instance(
        request_type="loglikelihood",
        doc={
            "array": np.array([1, 2], dtype=np.int64),
            "scalar": np.float32(0.25),
            "payload": b"cached-bytes",
            7: "integer-key",
            (1, "tuple"): "tuple-key",
            cache.TYPE_MARKER: "literal-marker-key",
        },
        arguments=(torch.tensor([1.5, 2.5], dtype=torch.float32),),
        idx=3,
        metadata=("task", 0, 1),
    )

    cache.save_to_cache("requests-json-types", [instance])

    cached = cache.load_from_cache("requests-json-types", cache=True)
    assert isinstance(cached, list)
    cached_instance = cached[0]
    assert isinstance(cached_instance, Instance)
    assert cached_instance.doc["array"] == [1, 2]
    assert cached_instance.doc["scalar"] == pytest.approx(0.25)
    assert cached_instance.doc["payload"] == b"cached-bytes"
    assert cached_instance.doc[7] == "integer-key"
    assert cached_instance.doc[(1, "tuple")] == "tuple-key"
    assert cached_instance.doc[cache.TYPE_MARKER] == "literal-marker-key"
    assert cached_instance.arguments == ([1.5, 2.5],)


def test_request_cache_roundtrips_pil_images(tmp_path, monkeypatch):
    image_module = pytest.importorskip("PIL.Image")
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    image = image_module.new("RGB", (2, 1), (12, 34, 56))
    instance = Instance(
        request_type="generate_until",
        doc={"question": "Q"},
        arguments=(image,),
        idx=5,
        metadata=("vision", 0, 1),
    )

    cache.save_to_cache("requests-json-image", [instance])

    cached = cache.load_from_cache("requests-json-image", cache=True)
    assert isinstance(cached, list)
    cached_instance = cached[0]
    assert isinstance(cached_instance, Instance)
    cached_image = cached_instance.arguments[0]
    assert cached_image.mode == "RGB"
    assert cached_image.size == (2, 1)
    assert cached_image.getpixel((0, 0)) == (12, 34, 56)


def test_request_cache_rejects_excessive_nesting(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "PATH", str(tmp_path))

    nested: list = []
    current = nested
    for _ in range(cache.MAX_JSON_NESTING_DEPTH + 1):
        child: list = []
        current.append(child)
        current = child

    with pytest.raises(ValueError, match="nesting exceeds"):
        cache.save_to_cache("requests-json-deep", nested)
