"""Each provider translates Attachment into its own native content parts."""

import base64

from coffee_with_llm import Attachment
from coffee_with_llm.providers.anthropic.messages_client import (
    _attachment_block,
    _user_content,
)
from coffee_with_llm.providers.google.text_client import _user_parts
from coffee_with_llm.providers.openai.responses_client import (
    _attachment_part as openai_part,
)
from coffee_with_llm.providers.openai.responses_client import _build_input_list

PDF = b"%PDF-1.4 fake"
PNG = b"\x89PNG\r\n\x1a\n fake"


def pdf(**kw):
    return Attachment(data=PDF, mime_type="application/pdf", **kw)


def png(**kw):
    return Attachment(data=PNG, mime_type="image/png", **kw)


class TestAnthropic:
    def test_pdf_becomes_base64_document_block(self):
        block = _attachment_block(pdf())
        assert block["type"] == "document"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "application/pdf"
        assert base64.b64decode(block["source"]["data"]) == PDF

    def test_image_becomes_image_block(self):
        assert _attachment_block(png())["type"] == "image"

    def test_filename_becomes_document_title(self):
        assert _attachment_block(pdf(filename="report.pdf"))["title"] == "report.pdf"

    def test_images_have_no_title_field(self):
        assert "title" not in _attachment_block(png(filename="shot.png"))

    def test_attachments_precede_prompt_text(self):
        """Anthropic documents PDFs-before-text as the higher-accuracy ordering."""
        content = _user_content("what is this?", [pdf()])
        assert [b["type"] for b in content] == ["document", "text"]
        assert content[-1]["text"] == "what is this?"

    def test_no_attachments_keeps_plain_string(self):
        assert _user_content("hello", None) == "hello"
        assert _user_content("hello", []) == "hello"


class TestOpenAI:
    def test_pdf_becomes_input_file_data_url(self):
        part = openai_part(pdf(filename="report.pdf"))
        assert part["type"] == "input_file"
        assert part["filename"] == "report.pdf"
        assert part["file_data"].startswith("data:application/pdf;base64,")

    def test_pdf_without_filename_gets_default(self):
        assert openai_part(pdf())["filename"] == "attachment.pdf"

    def test_image_becomes_input_image(self):
        part = openai_part(png())
        assert part["type"] == "input_image"
        assert part["image_url"].startswith("data:image/png;base64,")

    def test_input_list_attaches_to_prompt_turn(self):
        out = _build_input_list("q", None, [pdf()])
        assert len(out) == 1
        assert out[0]["role"] == "user"
        assert [p["type"] for p in out[0]["content"]] == ["input_file", "input_text"]

    def test_history_is_untouched_by_attachments(self):
        history = [{"role": "user", "content": "earlier"}]
        out = _build_input_list("q", history, [pdf()])
        assert out[0] == {"role": "user", "content": "earlier"}
        assert isinstance(out[1]["content"], list)

    def test_no_attachments_keeps_plain_string_content(self):
        out = _build_input_list("q", None, None)
        assert out[-1]["content"] == "q"


class TestGoogle:
    def test_pdf_becomes_inline_data_with_raw_bytes(self):
        """google-genai encodes on the way out; pre-encoding would double-encode."""
        parts = _user_parts("q", [pdf()])
        inline = parts[0]["inline_data"]
        assert inline["mime_type"] == "application/pdf"
        assert inline["data"] == PDF
        assert isinstance(inline["data"], bytes)

    def test_attachments_precede_text_part(self):
        parts = _user_parts("q", [pdf()])
        assert "inline_data" in parts[0]
        assert parts[-1] == {"text": "q"}

    def test_no_attachments_yields_text_only(self):
        assert _user_parts("q", None) == [{"text": "q"}]

    def test_multiple_attachments_preserve_order(self):
        parts = _user_parts("q", [pdf(), png()])
        mimes = [p["inline_data"]["mime_type"] for p in parts[:-1]]
        assert mimes == ["application/pdf", "image/png"]


class TestCrossProviderConsistency:
    """The same Attachment must reach every provider with identical bytes."""

    def test_payload_survives_all_three_translations(self):
        a = pdf(filename="doc.pdf")

        anthropic_bytes = base64.b64decode(_attachment_block(a)["source"]["data"])
        openai_bytes = base64.b64decode(openai_part(a)["file_data"].split(",", 1)[1])
        google_bytes = _user_parts("q", [a])[0]["inline_data"]["data"]

        assert anthropic_bytes == openai_bytes == google_bytes == PDF

    def test_every_provider_puts_attachments_before_text(self):
        a = pdf()
        assert [b["type"] for b in _user_content("q", [a])][-1] == "text"
        assert [p["type"] for p in _build_input_list("q", None, [a])[0]["content"]][-1] == (
            "input_text"
        )
        assert _user_parts("q", [a])[-1] == {"text": "q"}
