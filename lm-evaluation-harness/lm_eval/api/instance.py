from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


@dataclass
class Instance:
    CACHE_FIELDS: ClassVar[tuple[str, ...]] = (
        "request_type",
        "doc",
        "arguments",
        "idx",
        "metadata",
        "resps",
        "filtered_resps",
        "task_name",
        "doc_id",
        "repeats",
    )

    request_type: OutputType
    doc: dict
    arguments: tuple
    idx: int
    metadata: tuple[str | None, int | None, int | None] = field(
        default_factory=lambda: (None, None, None)
    )
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: str | None = None
    doc_id: int | None = None
    repeats: int | None = None

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata

    def to_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name) for field_name in self.CACHE_FIELDS
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Instance:
        task_name = data["task_name"]
        doc_id = data["doc_id"]
        repeats = data["repeats"]
        metadata = data["metadata"]
        if metadata != (task_name, doc_id, repeats):
            metadata = (task_name, doc_id, repeats)
        return cls(
            request_type=data["request_type"],
            doc=data["doc"],
            arguments=data["arguments"],
            idx=data["idx"],
            metadata=metadata,
            resps=data["resps"],
            filtered_resps=data["filtered_resps"],
        )

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )
