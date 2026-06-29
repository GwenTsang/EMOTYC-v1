from __future__ import annotations

from collections.abc import Callable, Sequence


TemplateFunction = Callable[[str], str]
EOS = "</s>"


def raw(text: str) -> str:
    """Return the text unchanged."""
    return text


def bca(text: str) -> str:
    """Format a text with before/current/after markers."""
    return f"before:{EOS}current: {text}{EOS}after:{EOS}"


def bca_with_context(text: str, before: str | None, after: str | None) -> str:
    """Format a text with adjacent before/current/after context."""
    previous = str(before) if before else EOS
    following = str(after) if after else EOS
    return f"before:{previous}{EOS}current: {text}{EOS}after:{following}{EOS}"


TEMPLATES: dict[str, TemplateFunction] = {
    "raw": raw,
    "bca": bca,
}


def apply_template(texts: Sequence[str], template: str, use_context: bool = False) -> list[str]:
    """Apply a named template to a sequence of texts."""
    try:
        formatter = TEMPLATES[template]
    except KeyError as exc:
        available = ", ".join(sorted(TEMPLATES))
        raise ValueError(f"Unknown template '{template}'. Available templates: {available}") from exc
    text_list = [str(text) for text in texts]
    if use_context:
        if template != "bca":
            raise ValueError("--use-context is only supported with template 'bca'")
        n_texts = len(text_list)
        return [
            bca_with_context(
                text_list[index],
                text_list[index - 1] if index > 0 else None,
                text_list[index + 1] if index < n_texts - 1 else None,
            )
            for index in range(n_texts)
        ]
    return [formatter(text) for text in text_list]
