"""Provider-agnostic binary attachments (documents and images).

An attachment is **input-side only**: the model reads it and answers in text.
Callers build one :class:`Attachment` regardless of provider; each provider
translates it into its own native content part, so the same calling code works
across Anthropic, OpenAI, and Google.

This module deliberately knows nothing about provider wire formats — that
knowledge lives in each provider package, so adding a provider does not mean
editing this file.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence, Union

from .exceptions import ValidationError

AttachmentKind = Literal["image", "document"]

#: MIME types every supported provider can read as an image.
IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }
)

#: MIME types every supported provider can read as a document.
DOCUMENT_MIME_TYPES: frozenset[str] = frozenset({"application/pdf"})

SUPPORTED_MIME_TYPES: frozenset[str] = IMAGE_MIME_TYPES | DOCUMENT_MIME_TYPES

# Providers cap the whole request, not the attachment (Anthropic 32MB,
# OpenAI/Google 50MB). Guard well below the smallest cap so oversized payloads
# fail locally with a clear message instead of as an opaque HTTP error.
MAX_ATTACHMENT_BYTES: int = 32 * 1024 * 1024


@dataclass(frozen=True)
class Attachment:
    """A binary input the model should read alongside the prompt.

    Args:
        data: Raw bytes. Never base64 — encoding is the provider's business.
        mime_type: e.g. ``"application/pdf"``, ``"image/png"``.
        filename: Optional display name. Some providers surface it to the model;
            those that don't simply ignore it.
    """

    data: bytes
    mime_type: str
    filename: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray)):
            raise ValidationError(
                f"Attachment.data must be bytes, got {type(self.data).__name__}"
            )
        if not self.data:
            raise ValidationError("Attachment.data must not be empty")
        if len(self.data) > MAX_ATTACHMENT_BYTES:
            raise ValidationError(
                f"Attachment is {len(self.data)} bytes, over the "
                f"{MAX_ATTACHMENT_BYTES}-byte limit; split it into smaller parts"
            )

        mime = (self.mime_type or "").strip().lower()
        if not mime:
            raise ValidationError("Attachment.mime_type is required")
        if mime not in SUPPORTED_MIME_TYPES:
            supported = ", ".join(sorted(SUPPORTED_MIME_TYPES))
            raise ValidationError(
                f"Unsupported attachment mime_type {self.mime_type!r}; supported: {supported}"
            )
        # Normalize onto the frozen instance.
        object.__setattr__(self, "mime_type", mime)

    @property
    def kind(self) -> AttachmentKind:
        """Whether providers should send this as an image or a document part."""
        return "image" if self.mime_type in IMAGE_MIME_TYPES else "document"

    def to_base64(self) -> str:
        """Base64-encoded payload (providers that take base64 rather than bytes)."""
        return base64.b64encode(self.data).decode("ascii")

    def to_data_url(self) -> str:
        """``data:`` URL form, used by providers that inline attachments as URLs."""
        return f"data:{self.mime_type};base64,{self.to_base64()}"

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        *,
        mime_type: Optional[str] = None,
    ) -> Attachment:
        """Read a file from disk, guessing the MIME type from its suffix."""
        file_path = Path(path)
        try:
            data = file_path.read_bytes()
        except OSError as e:
            raise ValidationError(f"Could not read attachment {file_path}: {e}") from e

        resolved = mime_type or mimetypes.guess_type(file_path.name)[0]
        if not resolved:
            raise ValidationError(
                f"Could not infer mime_type for {file_path.name}; pass mime_type explicitly"
            )
        return cls(data=data, mime_type=resolved, filename=file_path.name)


def normalize_attachments(
    attachments: Optional[Iterable[Attachment]],
) -> tuple[Attachment, ...]:
    """Validate and freeze caller-supplied attachments.

    Returns an empty tuple when ``attachments`` is ``None`` or empty, so
    providers can branch on truthiness without a ``None`` check.
    """
    if attachments is None:
        return ()
    if isinstance(attachments, (str, bytes, Attachment)):
        raise ValidationError(
            "attachments must be a sequence of Attachment objects, not a single value"
        )

    resolved = tuple(attachments)
    for index, item in enumerate(resolved):
        if not isinstance(item, Attachment):
            raise ValidationError(
                f"attachments[{index}] must be an Attachment, got {type(item).__name__}"
            )
    return resolved


def split_by_kind(
    attachments: Sequence[Attachment],
) -> tuple[tuple[Attachment, ...], tuple[Attachment, ...]]:
    """Partition into ``(images, documents)`` preserving caller order."""
    images = tuple(a for a in attachments if a.kind == "image")
    documents = tuple(a for a in attachments if a.kind == "document")
    return images, documents
