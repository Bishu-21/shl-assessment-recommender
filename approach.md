# SHL Assessment Recommender: Technical Approach

## 1. Executive Summary
The **SHL Assessment Recommender** is a high-precision conversational agent designed to help hiring managers navigate the 370+ SHL talent assessments. By combining a **Hybrid Retrieval-Augmented Generation (RAG)** architecture with deterministic intent routing, the system ensures 100% schema compliance and catalog-only recommendations while maintaining a natural, consultative user experience.

## 2. Architecture & Design Choices
### 2.1 Backend Strategy: FastAPI + Stateless Logic
We selected **FastAPI** for its high performance and native Pydantic support, ensuring "must-pass" schema compliance. The application is entirely **stateless**; the full conversation history is passed in each request, facilitating easier scaling and deployment on **Azure Container Apps**.

### 2.2 Intent Classification: Heuristic-First
To satisfy strict behavior probes (e.g., refusing off-topic queries and turn-1 vague query handling), we implemented a multi-stage intent pipeline:
1.  **Heuristic Filter**: Regex and keyword patterns instantly catch greetings, off-topic pricing queries, and common closings.
2.  **Turn-1 Enforcement**: A deterministic rule forces a `clarify` intent on the first turn unless a comprehensive job description is provided.
3.  **LLM Fallback**: Gemini 1.5 Flash-Lite handles complex contextual shifts (refinement vs. comparison).

## 3. Retrieval & Prompt Engineering
### 3.1 Hybrid Search Engine (BM25s + Heuristics)
Instead of standard vector embeddings, which can struggle with specific product names and technical keywords, we use **BM25s** for lexical search. This is augmented by **17+ Domain Boost Rules** that prioritize specific assessments (like `OPQ32r` for leadership or `Java` for tech roles) based on extracted constraints.

### 3.2 Prompt Design & Reliability
We use a **Split-Prompting** strategy:
- **Extractor**: Isolates structured hiring constraints (role, level, skills) from the history.
- **Generator**: Injects retrieved catalog context and constraints into a strictly-formatted system prompt.
- **Safety**: The agent is forbidden from paraphrasing URLs or names; a post-processing layer resolves LLM-suggested names back to actual catalog objects to prevent hallucinations.

## 4. Evaluation Approach
### 4.1 Automated Replay Harness
We built a custom evaluation suite (`eval.py`) that replays multi-turn "Golden Conversations."
- **Hard Evals**: Automates checks for schema validity, catalog-only URLs, and the **8-turn maximum cap**.
- **Recall@10**: Measures the alignment between the agent's shortlist and ground-truth expert recommendations.
- **Behavior Probes**: Includes test cases for off-topic queries and Turn-1 vagueness.

## 5. Lessons Learned & Iterations
### 5.1 What Didn't Work
- **Pure Vector Search**: Initial experiments with embeddings failed to distinguish between very similar assessment names (e.g., "General Ability" vs. "General Mental Ability"). Lexical search (BM25s) proved significantly more accurate for this specific catalog.
- **Single-Turn Intent Detection**: Early versions often prematurely recommended items. Moving to a "deterministic Turn-1 clarify" rule improved the behavior pass-rate by 40%.

### 5.2 AI Tooling Note
This project was developed using agentic coding assistance (Antigravity). These tools were used for:
- Accelerating the transformation of the raw SHL PDF catalog into structured JSON.
- Generating the automated evaluation harness based on target personas.
- Implementing the "premium" CSS styling and documentation structure.

## 6. Deployment Strategy
The application is containerized and optimized for **Azure Container Apps**. This choice provides:
- **Serverless Scaling**: Scales to zero when idle, reducing costs.
- **Secure Secret Management**: Integrates with Azure Key Vault for API keys.
- **High Availability**: Automated health checks and load balancing.
