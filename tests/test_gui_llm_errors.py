"""GUI behaviour when LLM briefing fails."""

from __future__ import annotations

import connect4_mcts.gui as gui
from connect4_mcts.game import Player


def test_rules_gate_inactive_after_briefing_failure() -> None:
    game = object.__new__(gui.HumanVsAgentGui)
    game.config = gui.GuiConfig(human=Player.RED, agent_name="llm")
    game.mode = "game"
    game.llm_awaiting_rules_ack = False
    game.llm_briefing_failed = True
    game.agent = type("Agent", (), {"rules_acknowledged": False})()

    assert not game._llm_rules_gate_active()


def test_report_llm_failure_sets_alert(monkeypatch) -> None:
    notified: list[tuple[bool, str, str]] = []

    def fake_notify(success: bool, title: str, text: str) -> None:
        notified.append((success, title, text))

    monkeypatch.setattr(gui, "_notify", fake_notify)

    game = object.__new__(gui.HumanVsAgentGui)
    game.agent = object()
    detail = game._report_llm_failure("LLM rules briefing failed", Exception("Error code: 429 - quota"))

    assert "quota" in detail.lower()
    assert game.llm_alert == detail
    assert notified == [(False, "LLM rules briefing failed", detail)]
