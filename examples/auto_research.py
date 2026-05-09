"""AutoResearch: Generate a survey paper on any topic using three-tier agents.

Usage:
    python examples/auto_research.py \
        --topic "KV cache optimization for multi-agent LLM inference" \
        --n-questions 20 \
        --backend anthropic:// \
        --output paper.md

Cost estimate (Sonnet workers, Opus planner/reducer):
    ~$0.80-1.20 for 20 questions at typical paper length

Time estimate:
    ~4-6 minutes for 20 questions with max_concurrent=10
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from batch_agent import BatchAgent, Tool


class ResearchPlan(BaseModel):
    items: list[str]
    scope: str


class ResearchAnswer(BaseModel):
    question: str
    answer: str
    key_findings: list[str]
    sources_consulted: list[str]


class SurveyPaper(BaseModel):
    title: str
    abstract: str
    sections: list[dict[str, Any]]
    conclusions: str
    open_questions: list[str]


async def main(topic: str, n: int, backend: str, output: str) -> None:
    print(f"Researching: {topic}")
    print(f"Questions: {n} | Backend: {backend}")
    print()

    results, paper = await BatchAgent.run_with_map_reduce(
        plan_prompt=(
            "You are a research director. Generate exactly {n} specific, "
            "focused research questions that together cover the topic: {topic}\n\n"
            "Each question should be answerable in 300-500 words by a researcher "
            "with web access. Questions should be non-overlapping and together "
            "give comprehensive coverage."
        ),
        plan_inputs={"topic": topic, "n": n},
        plan_output_schema=ResearchPlan,
        planner_model="claude-opus-4-6",
        task=(
            "You are a research analyst with web search access. "
            "Answer this research question thoroughly in 300-500 words: {item}\n\n"
            "Search for recent papers, blog posts, and benchmarks. "
            "Be specific; cite actual numbers and findings where available."
        ),
        output_schema=ResearchAnswer,
        tools=[Tool.web_search, Tool.read_file],
        worker_model="claude-sonnet-4-6",
        reduce=(
            "You are a senior researcher synthesizing a survey paper. "
            f"You have received {{n}} research answers on the topic: {topic}\n\n"
            "Write a comprehensive survey paper with: abstract, 4-6 themed sections "
            "that group related findings, conclusions, and open research questions. "
            "Be specific; include actual numbers and findings from the research answers."
        ),
        reduce_schema=SurveyPaper,
        reducer_model="claude-opus-4-6",
        model="claude-sonnet-4-6",
        backend=backend,
        max_concurrent=10,
        max_turns=4,
        on_result=lambda r: print(f"  {r.output.question[:60]}..." if r.ok else f"  FAILED: {r.error}"),
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# {paper.title}\n\n")
        f.write(f"## Abstract\n\n{paper.abstract}\n\n")
        for section in paper.sections:
            f.write(f"## {section['heading']}\n\n{section['content']}\n\n")
        f.write(f"## Conclusions\n\n{paper.conclusions}\n\n")
        f.write("## Open Questions\n\n")
        for question in paper.open_questions:
            f.write(f"- {question}\n")

    print(f"\nPaper written to {out}")
    print(f"Sections: {len(paper.sections)}")
    print(f"Research answers: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--backend", default="anthropic://")
    parser.add_argument("--output", default="paper.md")
    args = parser.parse_args()
    asyncio.run(main(args.topic, args.n_questions, args.backend, args.output))
