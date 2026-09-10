"""Root cause analysis via tree-of-thought beam search over the 5 Whys.

The 5 Whys is normally a single linear chain: ask "why did that happen" five
times in a row and take whatever answer falls out at the end. Here, at each
of the 5 levels, every surviving branch is expanded into several distinct
candidate answers to the next "why", each self-scored for plausibility.
Standard beam search — score = product of each step's confidence along the
path — keeps only the top BEAM_WIDTH branches (by that cumulative score) to
expand at the next level, instead of committing to a single guess per level.
After 5 levels, the single highest-scoring leaf's full why-chain is returned
as the root cause.

This module doesn't import or know about any specific vector store — the
caller (see main.py's root_cause_analysis tool) is expected to retrieve the
up-front evidence with this app's existing search tools and pass it in as
`evidence`. But a single up-front retrieval goes stale fast here: by level 3
or 4, the "why" being asked has drifted far from the original problem
statement, so evidence gathered for that original statement stops being
relevant — which is exactly what made root-cause answers read as generic
rather than grounded. To fix that, the caller also injects a
`retrieve_guidance` callback (query -> regulatory text), which this module
calls fresh at EVERY node with that node's own specific candidate cause as
the query — so each branch's rationale is grounded in guideline text
relevant to that specific branch, not just the original problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

BEAM_WIDTH = 3
BRANCHES_PER_NODE = 3
DEPTH = 5  # the 5 in "5 Whys"


@dataclass
class WhyNode:
    answer: str
    rationale: str
    confidence: float
    cumulative_score: float
    parent: Optional["WhyNode"] = None


def _ancestry(node: WhyNode) -> list[WhyNode]:
    """Why-answer nodes from level 1 up to and including `node`, oldest first.
    Excludes the synthetic root (whose 'answer' is just the problem statement)."""
    chain = []
    current = node
    while current.parent is not None:
        chain.append(current)
        current = current.parent
    return list(reversed(chain))


class WhyBranch(BaseModel):
    answer: str = Field(description=(
        "A specific, concrete candidate cause for why the previous answer "
        "happened — a mechanism, not a vague restatement (e.g. 'the bearing "
        "seized from loss of lubricant', not 'a failure occurred')."
    ))
    rationale: str = Field(description=(
        "Why this is plausible, citing the provided evidence (by ID or "
        "specific detail) where it supports this cause. If nothing in the "
        "evidence supports it, say so explicitly rather than implying support."
    ))
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Self-assessed plausibility (0.0-1.0) that this is a correct "
            "immediate cause given the evidence. Lower this when the answer "
            "is speculative rather than evidence-backed."
        ),
    )


class WhyBranches(BaseModel):
    branches: list[WhyBranch]


def _expand(
    llm: BaseChatModel,
    problem: str,
    evidence: str,
    node: WhyNode,
    retrieve_guidance: Callable[[str], str],
) -> list[WhyNode]:
    """Generate BRANCHES_PER_NODE candidate next-level why-answers for `node`,
    grounded in regulatory guidance retrieved fresh for `node.answer` itself
    (not just the original problem statement, which is what the static
    `evidence` was retrieved for and goes stale a few levels down)."""
    ancestry = _ancestry(node)
    chain_text = "\n".join(f"Why {i + 1}: {n.answer}" for i, n in enumerate(ancestry))
    guidance = retrieve_guidance(node.answer)

    structured_llm = llm.with_structured_output(WhyBranches)
    response: WhyBranches = structured_llm.invoke([
        SystemMessage(content=(
            "You are performing 5 Whys root cause analysis using tree-of-thought "
            f"reasoning. Propose exactly {BRANCHES_PER_NODE} distinct, specific, "
            "concrete candidate answers to the next 'why' question below. Each "
            "must be a genuinely different plausible mechanism — do not restate "
            "or trivially reword each other, and do not repeat a cause already "
            "given earlier in the chain. Ground answers in the evidence and "
            "regulatory guidance provided — cite specific guideline text "
            "(document/page) in the rationale when it supports a cause, rather "
            "than writing a generic mechanism unconnected to any retrieved "
            "material. You may still propose a cause the material doesn't "
            "cover, but must lower its confidence and say so explicitly."
        )),
        HumanMessage(content=(
            f"Problem: {problem}\n\n"
            f"Evidence gathered up front:\n{evidence or '(none retrieved)'}\n\n"
            f"Regulatory/safety guideline text retrieved specifically for this "
            f"branch's candidate cause ('{node.answer}'):\n"
            f"{guidance or '(no relevant guideline text found for this specific cause)'}\n\n"
            f"Reasoning so far:\n{chain_text or '(none yet — this is the first Why)'}\n\n"
            f"Why did '{node.answer}' happen?"
        )),
    ])

    return [
        WhyNode(
            answer=branch.answer,
            rationale=branch.rationale,
            confidence=branch.confidence,
            cumulative_score=node.cumulative_score * max(branch.confidence, 0.01),
            parent=node,
        )
        for branch in response.branches
    ]


def run_five_whys_beam_search(
    problem: str,
    evidence: str,
    llm: BaseChatModel,
    retrieve_guidance: Callable[[str], str],
) -> str:
    """Run a 5-level tree-of-thought beam search over the 5 Whys and return the
    most relevant root cause, with its full why-chain, as a formatted string.

    retrieve_guidance(query) -> text is called fresh at every node (not just
    once up front) with that node's own candidate cause as the query, so each
    level's regulatory grounding tracks what that specific branch is actually
    claiming rather than staying pinned to the original problem statement.
    """
    root = WhyNode(answer=problem, rationale="Starting problem statement.", confidence=1.0, cumulative_score=1.0)
    beam = [root]

    for _level in range(DEPTH):
        candidates = []
        for node in beam:
            candidates.extend(_expand(llm, problem, evidence, node, retrieve_guidance))
        candidates.sort(key=lambda n: n.cumulative_score, reverse=True)
        beam = candidates[:BEAM_WIDTH]

    best = max(beam, key=lambda n: n.cumulative_score)

    lines = [f"Problem: {problem}", ""]
    for i, node in enumerate(_ancestry(best), start=1):
        lines.append(f"Why {i}: {node.answer}")
        lines.append(f"  Rationale: {node.rationale} (confidence: {node.confidence:.2f})")
    lines.append("")
    lines.append(f"Root Cause: {best.answer}")
    lines.append(f"Overall path confidence: {best.cumulative_score:.3f}")
    return "\n".join(lines)
