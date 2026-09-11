"""LangGraph StateGraph that wires the AI Radar agents together.

Flow:
    START ─┬─> scout_frontier   ─┐
           ├─> scout_productie  ─┤
           ├─> scout_cases      ─┼─> synthesizer ─> critic ─┬─(approved)─> finalize ─> END
           └─> scout_hype       ─┘        ^                 │
                                          └──(revise)───────┘
                                                            finalize ─> rewriter ─> END

Usage:
    python graph.py                          # prints briefing + article
    python graph.py -o briefing.md -a artikel.md   # also writes them to files
"""

from __future__ import annotations

import argparse
from datetime import date
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents import (
    critic,
    finalize,
    rewriter,
    scout_cases,
    scout_frontier,
    scout_hype,
    scout_productie,
    synthesizer,
)
from state import RadarState


def route_after_critic(state: RadarState) -> Literal["finalize", "synthesizer"]:
    return "finalize" if state.get("approved") else "synthesizer"


def build_graph():
    builder = StateGraph(RadarState)

    # Nodes
    builder.add_node("scout_frontier", scout_frontier)
    builder.add_node("scout_productie", scout_productie)
    builder.add_node("scout_cases", scout_cases)
    builder.add_node("scout_hype", scout_hype)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("critic", critic)
    builder.add_node("finalize", finalize)
    builder.add_node("rewriter", rewriter)

    # Fan out: all scouts run in parallel from START
    for scout in ("scout_frontier", "scout_productie", "scout_cases", "scout_hype"):
        builder.add_edge(START, scout)
        builder.add_edge(scout, "synthesizer")  # fan in

    # Synthesize -> critique -> loop or finish
    builder.add_edge("synthesizer", "critic")
    builder.add_conditional_edges("critic", route_after_critic)
    builder.add_edge("finalize", "rewriter")
    builder.add_edge("rewriter", END)

    return builder.compile()


graph = build_graph()


def run(target_date: str | None = None) -> tuple[str, str]:
    """Return (briefing, article)."""
    initial: RadarState = {
        "date": target_date or date.today().isoformat(),
        "retries": 0,
    }
    result = graph.invoke(initial)
    return result["final_briefing"], result["article"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the AI Radar briefing.")
    parser.add_argument("-d", "--date", help="Date label (YYYY-MM-DD), default today")
    parser.add_argument("-o", "--output", help="Write briefing to this Markdown file")
    parser.add_argument("-a", "--article", help="Write the rewritten article to this file")
    args = parser.parse_args()

    briefing, article = run(args.date)
    print(briefing)
    print("\n\n" + "=" * 70 + "\nARTIKEL (concept)\n" + "=" * 70 + "\n")
    print(article)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(briefing)
        print(f"\nBriefing opgeslagen in {args.output}")
    if args.article:
        with open(args.article, "w", encoding="utf-8") as fh:
            fh.write(article)
        print(f"Artikel opgeslagen in {args.article}")
