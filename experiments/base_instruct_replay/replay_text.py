"""Render first-person five-failure replay prompts."""

from __future__ import annotations

from typing import Any


BOUNDARY = "My next guess will be:"


def game_intro() -> str:
    return (
        "I am playing four-digit Mastermind. Each guess contains exactly four "
        "digits from 0 to 9; repeated digits and a leading zero are allowed. "
        "After each guess, I receive two counts: digits in the correct position "
        "and additional correct digits in the wrong position. I use all earlier "
        "feedback when choosing each new guess."
    )


def intro(persona_text: str) -> str:
    return (
        "These are my broad personality tendencies:\n"
        f"{persona_text.strip()}\n\n"
        f"{game_intro()}\n\n"
        "My first guess will be:"
    )


def observed_guess(round_row: dict[str, Any]) -> str:
    guess = round_row.get("guess")
    if guess is None:
        return "[I failed to provide a valid four-digit guess.]"
    return str(guess)


def feedback_text(round_row: dict[str, Any]) -> str:
    round_index = int(round_row["round"])
    guess = round_row.get("guess")
    feedback = round_row.get("feedback")
    if guess is None:
        event = "That counted as an unsuccessful turn without game feedback."
    else:
        event = (
            f"The feedback was {feedback[0]} digit(s) in "
            f"the correct position and {feedback[1]} additional digit(s) correct "
            "but in the wrong position."
        )
    return (
        f"Round {round_index}. {event}\n"
        f"I have made {round_index} consecutive unsuccessful guess(es).\n"
        f"{BOUNDARY}"
    )


def trajectory_texts(persona_text: str, path: dict[str, Any]) -> tuple[str, list[str]]:
    """Build baseline and five-round replay prompts ending at each generation boundary."""
    prefix = intro(persona_text)
    history = prefix
    texts: list[str] = []
    for row in path["rounds"]:
        history += " " + observed_guess(row)
        history += "\n\n" + feedback_text(row)
        texts.append(history)
    return prefix, texts
