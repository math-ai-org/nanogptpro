import base64
import hashlib
import io
import json
import logging
import os
import re
from contextlib import suppress
from typing import Any

from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

OVERRIDE_PATH = os.getenv("LM_HARNESS_CACHE_PATH")


PATH = OVERRIDE_PATH or f"{MODULE_DIR}/.cache"

# This should be sufficient for uniqueness
HASH_INPUT = "EleutherAI-lm-evaluation-harness"

HASH_PREFIX = hashlib.sha256(HASH_INPUT.encode("utf-8")).hexdigest()

FILE_SUFFIX = f".{HASH_PREFIX}.json"
TYPE_MARKER = "__lm_eval_cache_type__"

# Keep cache file basenames comfortably below common 255-byte filesystem limits.
# Some request cache keys include model paths or endpoint identifiers, which can be
# very long. Preserve a readable prefix and append a hash of the full key so cache
# entries remain stable and unique without overflowing the filesystem limit.
MAX_CACHE_FILENAME_BYTES = 240
CACHE_KEY_HASH_LEN = 16
MAX_JSON_NESTING_DEPTH = 100


def _check_json_depth(depth: int) -> None:
    if depth > MAX_JSON_NESTING_DEPTH:
        raise ValueError(
            f"Cache payload nesting exceeds {MAX_JSON_NESTING_DEPTH} levels."
        )


def _to_pil_image_payload(obj: Any) -> dict[str, str] | None:
    with suppress(ImportError):
        from PIL import Image

        if isinstance(obj, Image.Image):
            buffer = io.BytesIO()
            obj.save(buffer, format="PNG")
            return {
                TYPE_MARKER: "PIL.Image",
                "format": "PNG",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
    return None


def _from_pil_image_payload(obj: dict[str, Any]) -> Any:
    from PIL import Image

    payload = obj["data"]
    image_bytes = base64.b64decode(payload.encode("ascii"))
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.copy()


def _to_list_payload(obj: Any, depth: int) -> Any:
    detach = getattr(obj, "detach", None)
    if callable(detach):
        obj = detach()
    cpu = getattr(obj, "cpu", None)
    if callable(cpu):
        obj = cpu()
    tolist = getattr(obj, "tolist", None)
    if not callable(tolist):
        return None
    return _to_jsonable(tolist(), depth)


def _cache_file_path(file_name: str) -> str:
    safe_file_name = re.sub(r"[/\\\\]+", "_", file_name)
    basename = f"{safe_file_name}{FILE_SUFFIX}"

    if len(basename.encode("utf-8")) <= MAX_CACHE_FILENAME_BYTES:
        return os.path.join(PATH, basename)

    file_name_hash = hashlib.sha256(file_name.encode("utf-8")).hexdigest()[
        :CACHE_KEY_HASH_LEN
    ]
    hashed_suffix = f"-{file_name_hash}{FILE_SUFFIX}"
    max_prefix_bytes = MAX_CACHE_FILENAME_BYTES - len(hashed_suffix.encode("utf-8"))

    prefix = safe_file_name.encode("utf-8")[:max_prefix_bytes]
    prefix = prefix.decode("utf-8", errors="ignore").rstrip("-_.") or "cache"

    return os.path.join(PATH, f"{prefix}{hashed_suffix}")


def _to_jsonable(obj: Any, depth: int = 0) -> Any:
    _check_json_depth(depth)
    next_depth = depth + 1
    pil_payload = _to_pil_image_payload(obj)
    if pil_payload is not None:
        return pil_payload
    if isinstance(obj, Instance):
        instance_payload = obj.to_dict()
        return {
            TYPE_MARKER: "Instance",
            **{
                field_name: _to_jsonable(instance_payload[field_name], next_depth)
                for field_name in Instance.CACHE_FIELDS
            },
        }
    if isinstance(obj, tuple):
        return {
            TYPE_MARKER: "tuple",
            "items": [_to_jsonable(item, next_depth) for item in obj],
        }
    if isinstance(obj, list):
        return [_to_jsonable(item, next_depth) for item in obj]
    if isinstance(obj, dict):
        if TYPE_MARKER in obj or any(not isinstance(key, str) for key in obj):
            return {
                TYPE_MARKER: "dict",
                "items": [
                    [_to_jsonable(key, next_depth), _to_jsonable(value, next_depth)]
                    for key, value in obj.items()
                ],
            }
        return {str(key): _to_jsonable(value, next_depth) for key, value in obj.items()}
    if isinstance(obj, bytes):
        return {
            TYPE_MARKER: "bytes",
            "data": base64.b64encode(obj).decode("ascii"),
        }
    list_payload = _to_list_payload(obj, next_depth)
    if list_payload is not None:
        return list_payload
    return obj


def _from_jsonable(obj: Any, depth: int = 0) -> Any:
    _check_json_depth(depth)
    next_depth = depth + 1
    if isinstance(obj, list):
        return [_from_jsonable(item, next_depth) for item in obj]
    if not isinstance(obj, dict):
        return obj

    marker = obj.get(TYPE_MARKER)
    if marker == "tuple":
        return tuple(_from_jsonable(item, next_depth) for item in obj["items"])
    if marker == "dict":
        return {
            _from_jsonable(key, next_depth): _from_jsonable(value, next_depth)
            for key, value in obj["items"]
        }
    if marker == "bytes":
        return base64.b64decode(obj["data"].encode("ascii"))
    if marker == "PIL.Image":
        return _from_pil_image_payload(obj)
    if marker == "Instance":
        instance_payload = {
            field_name: _from_jsonable(obj[field_name], next_depth)
            for field_name in Instance.CACHE_FIELDS
        }
        return Instance.from_dict(instance_payload)

    return {key: _from_jsonable(value, next_depth) for key, value in obj.items()}


def load_from_cache(file_name: str, cache: bool = False) -> Any | None:
    if not cache:
        return
    try:
        path = _cache_file_path(file_name)

        with open(path, encoding="utf-8") as file:
            cached_task_dict = json.load(file)
            return _from_jsonable(cached_task_dict)

    except (
        ImportError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        eval_logger.debug("%s is not cached, generating...", file_name)


def save_to_cache(file_name: str, obj: Any) -> None:
    os.makedirs(PATH, exist_ok=True)

    file_path = _cache_file_path(file_name)

    eval_logger.debug("Saving %s to cache...", file_path)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(_to_jsonable(obj), file)


# NOTE the "key" param is to allow for flexibility
def delete_cache(key: str = ""):
    files = os.listdir(PATH)

    for file in files:
        if file.startswith(key) and file.endswith(FILE_SUFFIX):
            file_path = f"{PATH}/{file}"
            os.unlink(file_path)
