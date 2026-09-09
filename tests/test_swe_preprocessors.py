# SPDX-License-Identifier: Apache-2.0

from examples.swe.preprocessors import StripOpenCodeEnvironmentDate


def _opencode_system_prompt(date: str) -> str:
    return (
        "You are powered by the configured model.\n"
        "<env>\n"
        " Working directory: /repo\n"
        " Platform: linux\n"
        f" Today's date: {date}\n"
        "</env>"
    )


def test_strip_opencode_environment_date_stabilizes_system_prompt():
    preprocessor = StripOpenCodeEnvironmentDate()
    before_midnight = [
        {"role": "system", "content": _opencode_system_prompt("Tue Sep 08 2026")}
    ]
    after_midnight = [
        {"role": "system", "content": _opencode_system_prompt("Wed Sep 09 2026")}
    ]

    assert preprocessor(before_midnight) == preprocessor(after_midnight)
    assert "Today's date:" not in before_midnight[0]["content"]
    assert before_midnight[0]["content"].endswith(" Platform: linux\n</env>")


def test_strip_opencode_environment_date_preserves_other_date_text():
    preprocessor = StripOpenCodeEnvironmentDate()
    messages = [
        {"role": "system", "content": "Today's date: Wed Sep 09 2026"},
        {"role": "user", "content": _opencode_system_prompt("Wed Sep 09 2026")},
    ]

    preprocessor(messages)

    assert messages[0]["content"] == "Today's date: Wed Sep 09 2026"
    assert "Today's date: Wed Sep 09 2026" in messages[1]["content"]
