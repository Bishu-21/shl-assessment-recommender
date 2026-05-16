"""
Hybrid retrieval engine for SHL assessments.

Combines BM25 text search with heuristic domain boosts
to surface the most relevant assessments for a given query.
"""

from __future__ import annotations

from dataclasses import dataclass

import bm25s
import numpy as np

from .catalog import Catalog, CatalogItem, catalog


# ---------------------------------------------------------------------------
# Domain-specific heuristic boost rules
# ---------------------------------------------------------------------------
@dataclass
class BoostRule:
    """If *any* trigger word appears in the query, boost the named assessments."""

    triggers: list[str]
    boost_names: list[str]  # substrings matched case-insensitively against item.name
    boost_score: float = 5.0


BOOST_RULES: list[BoostRule] = [
    # Java / Backend Engineering
    BoostRule(
        triggers=["java", "spring", "microservice", "backend", "rest api", "restful", "j2ee", "hibernate", "maven"],
        boost_names=[
            "core java",
            "java 8",
            "java 11",
            "spring",
            "restful web services",
            "sql",
            "docker",
            "verify interactive g+",
            "opq32r",
            "smart interview live coding",
        ],
        boost_score=8.0,
    ),
    # Frontend / Web Development
    BoostRule(
        triggers=["frontend", "front end", "react", "angular", "vue", "javascript", "html", "css", "web development", "ui/ux"],
        boost_names=[
            "automata front end",
            "javascript",
            "html5",
            "css3",
            "reactjs",
            "angular",
            "verify interactive g+",
            "opq32r",
        ],
        boost_score=8.0,
    ),
    # Senior / Leadership / Management
    BoostRule(
        triggers=["senior", "leadership", "director", "executive", "cxo", "vp", "c-level", "c-suite", "management", "manager", "head of"],
        boost_names=[
            "opq32r",
            "opq leadership report",
            "opq universal competency",
            "leadership report",
            "verify interactive g+",
            "motivation questionnaire",
            "graduate scenarios",
            "scenarios",
            "management simulation",
        ],
        boost_score=7.0,
    ),
    # Graduate / Entry-level / Management Trainee
    BoostRule(
        triggers=["graduate", "trainee", "entry-level", "entry level", "campus", "fresher", "recent graduate", "intern", "management trainee"],
        boost_names=[
            "graduate scenarios",
            "verify interactive g+",
            "opq32r",
            "verify interactive – numerical",
            "verify interactive – inductive",
            "general ability",
            "motivation questionnaire",
        ],
        boost_score=10.0,
    ),
    # Customer Service / Contact Center / Retail / Sales Associate
    BoostRule(
        triggers=[
            "customer service",
            "contact center",
            "contact centre",
            "call center",
            "call centre",
            "inbound calls",
            "customer support",
            "retail",
            "sales associate",
            "cashier",
            "store manager",
            "shop",
            "store",
        ],
        boost_names=[
            "retail sales and service simulation",
            "entry level customer serv-retail & contact center",
            "customer service phone simulation",
            "contact center call simulation",
            "svar",
            "situational judgment",
            "opq32r",
            "global skills assessment",
        ],
        boost_score=10.0,
    ),
    # Project Management / Agile
    BoostRule(
        triggers=["project manager", "project management", "pmp", "agile", "scrum", "scrum master", "kanban", "product manager"],
        boost_names=[
            "project management",
            "agile",
            "opq32r",
            "situational judgment",
            "verify interactive g+",
        ],
        boost_score=7.0,
    ),
    # Safety / Manufacturing / Industrial
    BoostRule(
        triggers=[
            "safety",
            "manufacturing",
            "industrial",
            "plant operator",
            "chemical",
            "dependability",
            "reliability",
            "warehouse",
            "logistics",
            "forklift",
        ],
        boost_names=[
            "dependability and safety instrument",
            "safety & dependability",
            "safety and dependability",
            "workplace health and safety",
            "mechanical comprehension",
            "opq32r",
        ],
        boost_score=8.0,
    ),
    # Finance / Accounting
    BoostRule(
        triggers=[
            "finance",
            "financial",
            "accounting",
            "accounts payable",
            "accounts receivable",
            "bookkeeping",
            "audit",
            "tax",
            "bank",
            "banking",
        ],
        boost_names=[
            "financial accounting",
            "accounts payable",
            "accounts receivable",
            "verify interactive – numerical",
            "verify interactive g+",
            "basic statistics",
            "opq32r",
        ],
        boost_score=7.0,
    ),
    # Marketing / PR
    BoostRule(
        triggers=["marketing", "branding", "advertisement", "pr", "public relations", "market research", "social media"],
        boost_names=[
            "marketing",
            "opq32r",
            "verify interactive g+",
            "situational judgment",
        ],
        boost_score=7.0,
    ),
    # Supply Chain / Logistics
    BoostRule(
        triggers=["supply chain", "logistics", "shipping", "procurement", "inventory", "distribution"],
        boost_names=[
            "supply chain management",
            "inventory management",
            "logistics",
            "opq32r",
            "verify interactive g+",
        ],
        boost_score=7.0,
    ),
    # Personality / Culture (Generic)
    BoostRule(
        triggers=["personality", "behaviour", "behavior", "cultural fit", "fit", "soft skills", "opq", "attributes"],
        boost_names=[
            "opq32r",
            "opq universal competency",
            "opq leadership report",
            "motivation questionnaire",
        ],
        boost_score=7.0,
    ),
    # Sales / Commercial
    BoostRule(
        triggers=["sales", "selling", "commercial", "revenue", "business development", "account manager", "b2b", "b2c"],
        boost_names=[
            "opq mq sales report",
            "sales transformation",
            "global skills assessment",
            "opq32r",
            "motivation questionnaire",
        ],
        boost_score=7.0,
    ),
    # Admin / Office / Support
    BoostRule(
        triggers=["admin", "assistant", "administrative", "office", "clerical", "secretary", "receptionist", "data entry"],
        boost_names=[
            "ms excel",
            "ms word",
            "microsoft excel",
            "microsoft word",
            "typing",
            "data entry",
            "opq32r",
            "verify interactive – numerical",
        ],
        boost_score=7.0,
    ),
    # Healthcare / Medical
    BoostRule(
        triggers=["healthcare", "medical", "hospital", "clinical", "hipaa", "patient", "nursing", "health care", "doctor", "pharmacy"],
        boost_names=[
            "hipaa",
            "medical terminology",
            "dependability and safety instrument",
            "opq32r",
            "situational judgment",
        ],
        boost_score=7.0,
    ),
    # Python / Data Science / Analytics
    BoostRule(
        triggers=["python", "data science", "machine learning", "ml", "analytics", "data engineer", "statistics", "pandas", "numpy"],
        boost_names=[
            "python",
            "basic statistics",
            "sql",
            "verify interactive g+",
            "apache spark",
            "automata data science",
        ],
        boost_score=8.0,
    ),
    # DevOps / Cloud / Infrastructure / Security
    BoostRule(
        triggers=["devops", "cloud", "aws", "azure", "docker", "kubernetes", "ci/cd", "infrastructure", "linux", "cybersecurity", "security"],
        boost_names=[
            "amazon web services",
            "docker",
            "linux",
            "kubernetes",
            "smart interview live coding",
            "verify interactive g+",
            "cyber security",
        ],
        boost_score=8.0,
    ),
    # Specific Technical Tools (Selenium, RPA)
    BoostRule(
        triggers=["selenium", "automation testing", "qa automation", "rpa", "uipath", "automation anywhere"],
        boost_names=[
            "automata selenium",
            "automation anywhere rpa development",
            "software testing",
            "verify interactive g+",
        ],
        boost_score=9.0,
    ),
    # Cognitive / Reasoning (Generic)
    BoostRule(
        triggers=["cognitive", "reasoning", "aptitude", "numerical", "verbal", "deductive", "inductive", "logic", "intelligence"],
        boost_names=[
            "verify interactive g+",
            "verify interactive – numerical",
            "verify interactive – deductive",
            "verify interactive – inductive",
            "verify numerical ability",
            "verify verbal ability",
        ],
        boost_score=7.0,
    ),
]


class Retriever:
    """Hybrid BM25 + heuristic retrieval engine."""

    def __init__(self, cat: Catalog | None = None) -> None:
        self._catalog = cat or catalog
        self._bm25: bm25s.BM25 | None = None
        self._corpus_texts: list[str] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------
    def build_index(self) -> None:
        """Build BM25 index from the catalog."""
        self._corpus_texts = [
            self._catalog.search_text(item)
            for item in self._catalog.items
        ]
        # pyrefly: ignore [bad-argument-type]
        corpus_tokens = bm25s.tokenize(self._corpus_texts, stemmer=None)
        self._bm25 = bm25s.BM25(method="robertson")
        self._bm25.index(corpus_tokens)
        print(f"[retriever] BM25 index built over {len(self._corpus_texts)} documents")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 15,
        min_score: float = 0.0,
    ) -> list[tuple[CatalogItem, float]]:
        """
        Retrieve top-K catalog items for a query.

        Returns list of (CatalogItem, score) tuples sorted by score descending.
        """
        if not self._bm25:
            self.build_index()

        assert self._bm25 is not None

        # BM25 scores — retrieve all docs so we can apply heuristic boosts
        n_docs = len(self._catalog.items)
        # pyrefly: ignore [bad-argument-type]
        query_tokens = bm25s.tokenize([query], stemmer=None)
        # Get all results so we have full score vector for boosting
        results_arr, scores_arr = self._bm25.retrieve(
            query_tokens, k=min(n_docs, top_k * 3)
        )

        # Build a full score array, fill in BM25 scores at retrieved positions
        scores = np.zeros(n_docs, dtype=np.float64)
        for i in range(results_arr.shape[1]):
            doc_idx = int(results_arr[0, i])
            scores[doc_idx] = float(scores_arr[0, i])

        # Apply heuristic boosts
        query_lower = query.lower()
        for rule in BOOST_RULES:
            if any(trigger in query_lower for trigger in rule.triggers):
                for idx, item in enumerate(self._catalog.items):
                    item_name_lower = item.name.lower()
                    for boost_name in rule.boost_names:
                        if boost_name in item_name_lower:
                            scores[idx] += rule.boost_score
                            break

        # Rank and filter
        ranked = np.argsort(-scores)

        results: list[tuple[CatalogItem, float]] = []
        for idx in ranked:
            s = scores[idx]
            if s <= min_score:
                continue
            results.append((self._catalog.items[int(idx)], float(s)))
            if len(results) >= top_k:
                break

        return results


# Singleton
retriever = Retriever()
