from __future__ import annotations

from calls_analyser.services.prompt import PromptService, PromptTemplate


def make_service() -> PromptService:
    return PromptService(
        {
            "simple": PromptTemplate(key="simple", title="Simple", body="Default"),
            "detailed": PromptTemplate(key="detailed", title="Detailed", body="More info"),
        }
    )


def test_get_prompt_returns_template_when_present() -> None:
    service = make_service()

    template = service.get_prompt("detailed")

    assert template.title == "Detailed"
    assert template.body == "More info"


def test_get_prompt_returns_fallback_when_missing() -> None:
    service = make_service()

    template = service.get_prompt("missing")

    assert template.key == "simple"


def test_get_prompt_uses_custom_fallback_key() -> None:
    service = make_service()

    template = service.get_prompt("missing", fallback_key="detailed")

    assert template.key == "detailed"


def test_list_templates_returns_copy() -> None:
    service = make_service()

    templates = service.list_templates()
    templates.pop("simple")

    # Original storage should remain intact
    assert service.get_prompt("simple").key == "simple"
    assert set(service.list_templates()) == {"simple", "detailed"}
