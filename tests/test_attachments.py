"""Tests for provider-agnostic attachments."""

import base64

import pytest

from coffee_with_llm import Attachment
from coffee_with_llm.attachments import (
    MAX_ATTACHMENT_BYTES,
    normalize_attachments,
    split_by_kind,
)
from coffee_with_llm.exceptions import ValidationError

PDF = b"%PDF-1.4 fake"
PNG = b"\x89PNG\r\n\x1a\n fake"


class TestAttachmentValidation:
    def test_rejects_empty_data(self):
        with pytest.raises(ValidationError, match="must not be empty"):
            Attachment(data=b"", mime_type="application/pdf")

    def test_rejects_non_bytes_data(self):
        with pytest.raises(ValidationError, match="must be bytes"):
            Attachment(data="not bytes", mime_type="application/pdf")

    def test_rejects_missing_mime_type(self):
        with pytest.raises(ValidationError, match="mime_type is required"):
            Attachment(data=PDF, mime_type="")

    def test_rejects_unsupported_mime_type(self):
        with pytest.raises(ValidationError, match="Unsupported attachment mime_type"):
            Attachment(data=b"MZ", mime_type="application/x-msdownload")

    def test_rejects_oversized_payload(self):
        with pytest.raises(ValidationError, match="over the"):
            Attachment(data=b"x" * (MAX_ATTACHMENT_BYTES + 1), mime_type="application/pdf")

    def test_normalizes_mime_case_and_whitespace(self):
        a = Attachment(data=PDF, mime_type="  APPLICATION/PDF  ")
        assert a.mime_type == "application/pdf"


class TestAttachmentEncoding:
    def test_kind_classification(self):
        assert Attachment(data=PDF, mime_type="application/pdf").kind == "document"
        assert Attachment(data=PNG, mime_type="image/png").kind == "image"

    def test_to_base64_roundtrip(self):
        a = Attachment(data=PDF, mime_type="application/pdf")
        assert base64.b64decode(a.to_base64()) == PDF

    def test_to_data_url(self):
        a = Attachment(data=PNG, mime_type="image/png")
        assert a.to_data_url().startswith("data:image/png;base64,")
        payload = a.to_data_url().split(",", 1)[1]
        assert base64.b64decode(payload) == PNG

    def test_data_is_not_pre_encoded(self):
        """`data` stays raw so providers taking bytes don't double-encode."""
        a = Attachment(data=PDF, mime_type="application/pdf")
        assert a.data == PDF


class TestFromPath:
    def test_reads_and_infers_mime(self, tmp_path):
        p = tmp_path / "doc.pdf"
        p.write_bytes(PDF)
        a = Attachment.from_path(p)
        assert a.data == PDF
        assert a.mime_type == "application/pdf"
        assert a.filename == "doc.pdf"

    def test_explicit_mime_overrides_suffix(self, tmp_path):
        p = tmp_path / "image.bin"
        p.write_bytes(PNG)
        assert Attachment.from_path(p, mime_type="image/png").mime_type == "image/png"

    def test_unknown_suffix_raises(self, tmp_path):
        p = tmp_path / "mystery.zzz"
        p.write_bytes(PDF)
        with pytest.raises(ValidationError, match="Could not infer mime_type"):
            Attachment.from_path(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValidationError, match="Could not read attachment"):
            Attachment.from_path(tmp_path / "nope.pdf")


class TestNormalize:
    def test_none_becomes_empty_tuple(self):
        assert normalize_attachments(None) == ()

    def test_passes_through_valid(self):
        a = Attachment(data=PDF, mime_type="application/pdf")
        assert normalize_attachments([a]) == (a,)

    def test_rejects_bare_attachment(self):
        a = Attachment(data=PDF, mime_type="application/pdf")
        with pytest.raises(ValidationError, match="sequence of Attachment"):
            normalize_attachments(a)

    def test_rejects_wrong_element_type(self):
        with pytest.raises(ValidationError, match=r"attachments\[0\] must be an Attachment"):
            normalize_attachments(["/path/to/file.pdf"])

    def test_split_by_kind_preserves_order(self):
        img1 = Attachment(data=PNG, mime_type="image/png")
        doc = Attachment(data=PDF, mime_type="application/pdf")
        img2 = Attachment(data=PNG, mime_type="image/jpeg")
        images, documents = split_by_kind([img1, doc, img2])
        assert images == (img1, img2)
        assert documents == (doc,)
