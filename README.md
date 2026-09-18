# MOSAIC — Multi-Agent Clinical Trial Intelligence System

> A production-oriented multi-agent AI system that analyzes clinical-trial and research-literature evidence to surface potential research-integrity signals using six specialist agents, persistent memory, human review, and cloud-native infrastructure.

**ClinicalTrials.gov + PubMed → Evidence Processing → Persistent Memory → Parallel Agents → Human Review → Intelligence**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.6-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue.svg)](https://www.postgresql.org/)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-blue.svg)](https://cloud.google.com/run)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Table of Contents

* [At a Glance](#-at-a-glance)
* [The Problem](#-the-problem)
* [What I Built](#-what-i-built)
* [How MOSAIC Works](#-how-mosaic-works)
* [Product Walkthrough](#-product-walkthrough)
* [System Architecture](#-system-architecture)
* [Data and Knowledge Pipeline](#-data-and-knowledge-pipeline)
* [Multi-Agent Reasoning](#-multi-agent-reasoning)
* [Memory Architecture](#-memory-architecture)
* [Human-in-the-Loop Learning](#-human-in-the-loop-learning)
* [Database Design](#-database-design)
* [API Architecture](#-api-architecture)
* [Key Engineering Decisions](#-key-engineering-decisions)
* [Challenges and Solutions](#-challenges-and-solutions)
* [Evaluation](#-evaluation)
* [Deployment](#-deployment)
* [Project Structure](#-project-structure)
* [Getting Started](#-getting-started)
* [API Usage](#-api-usage)
* [Testing](#-testing)
* [Limitations and Future Work](#-limitations-and-future-work)
* [My Contributions](#-my-contributions)
* [License](#-license)

---

# 🎯 At a Glance

|                    |                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| **Problem**        | Research-integrity signals are distributed across clinical-trial records and research literature |
| **Data Sources**   | ClinicalTrials.gov and PubMed                                                                    |
| **Core Approach**  | Multi-agent investigation over structured and semantic evidence                                  |
| **Agents**         | 6 specialist agents + supervisor                                                                 |
| **Execution**      | Parallel agent execution using LangGraph                                                         |
| **Memory**         | Episodic, procedural, and semantic                                                               |
| **Human Feedback** | Human review decisions feed procedural memory                                                    |
| **Vector Search**  | PostgreSQL + pgvector                                                                            |
| **API**            | FastAPI                                                                                          |
| **Deployment**     | Google Cloud Run                                                                                 |
| **Storage**        | Google Cloud Storage + Cloud SQL                                                                 |
| **LLM**            | GPT-4o                                                                                           |
| **Embeddings**     | OpenAI `text-embedding-3-small`                                                                  |
| **Validation**     | Pydantic v2                                                                                      |
| **Retry Handling** | Tenacity                                                                                         |

### Key Capabilities

* 🔎 Detect potential research-integrity signals across studies
* 🤖 Run six specialist investigations in parallel
* 🧠 Maintain persistent agent memory across sessions
* 👤 Route uncertain findings through human review
* 🔄 Learn from reviewer corrections through procedural memory
* 📚 Maintain accumulated sponsor-level intelligence
* 🗃️ Combine structured PostgreSQL queries with semantic vector search
* 🚀 Expose the system through a REST API
* ☁️ Deploy the API on Google Cloud Run

---

# 🔍 The Problem

Clinical-trial information is distributed across large public datasets and research literature.

A researcher investigating a trial may need to connect information such as:

* trial completion status
* expected result-posting status
* stated outcomes
* timelines
* sponsor history
* published research
* safety information
* patterns across related studies

The challenge is not simply retrieving a single document.

The challenge is **connecting evidence across multiple sources and multiple studies**.

MOSAIC is designed around this problem:

> **Can multiple specialized AI investigators examine different dimensions of clinical-trial evidence, combine their findings, and continuously improve through human feedback?**

The system currently integrates ClinicalTrials.gov and PubMed as its primary external research sources.

---

# 💡 What I Built

MOSAIC is a multi-agent clinical-trial intelligence system.

Instead of asking a single LLM to perform every investigation, MOSAIC decomposes the analysis into specialized investigations.

A typical analysis follows this process:

```text
User Investigation
        │
        ▼
    Supervisor
        │
        ▼
 ┌──────┼──────┬──────┬──────┬──────┐
 ▼      ▼      ▼      ▼      ▼      ▼
Missing Broken Track Pattern Side Timeline
Results Promises Record Finder Effect Analyst
 └──────┴──────┴──────┴──────┴──────┘
                 │
                 ▼
          Evidence-backed
             Signals
                 │
                 ▼
             HITL Gate
                 │
          ┌──────┴──────┐
          ▼             ▼
       Accepted       Review
                        │
                        ▼
                Procedural Memory
```

Each agent has a focused responsibility, while the supervisor coordinates the overall investigation and compiles the final intelligence brief.

---

# 🔄 How MOSAIC Works

## End-to-End Flow

```text
                  ┌─────────────────────┐
                  │   ClinicalTrials.gov│
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐
                  │       PubMed        │
                  └──────────┬──────────┘
                             │
                             ▼
                   ┌─────────────────┐
                   │    Ingestion    │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │   Processing    │
                   │ Chunk + Embed   │
                   └────────┬────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ PostgreSQL +        │
                 │ pgvector            │
                 └──────────┬──────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Supervisor  │
                    └──────┬───────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
 Missing Results      Broken Promises      Track Record
       │                   │                   │
       ├─────────────┬─────┴─────┬─────────────┤
       ▼             ▼           ▼             ▼
 Pattern Finder  Side Effects  Timeline     Evidence
       │             │           │
       └─────────────┴───────────┘
                     │
                     ▼
               ┌────────────┐
               │  HITL Gate │
               └─────┬──────┘
                     │
              ┌──────┴───────┐
              ▼              ▼
         High Confidence   Low Confidence
              │              │
              ▼              ▼
           Signals      Review Queue
                             │
                             ▼
                       Human Decision
                             │
                             ▼
                     Procedural Memory
```

The important architectural property is that the specialist investigations are **independent and can execute in parallel**, rather than forcing every investigation into a sequential chain.

---

# 🖥️ Product Walkthrough

## 1. Submit an Investigation

The system accepts a natural-language investigation task.

Example:

```json
{
  "task": "Find completed clinical trials where results were never posted",
  "max_studies": 3
}
```

---

## 2. Supervisor Coordinates the Investigation

The supervisor determines which specialist investigations are relevant and coordinates the agent graph.

Rather than implementing one large agent containing every rule, MOSAIC separates responsibilities across specialist agents.

---

## 3. Specialist Agents Investigate

The active agents query available structured data, semantic evidence, external sources, and persistent memory.

For example:

```text
Missing Results Agent
        │
        └── Checks completion + result-posting information

Track Record Agent
        │
        └── Checks historical sponsor behaviour

Pattern Finder Agent
        │
        └── Searches for cross-study relationships
```

---

## 4. Signals Are Generated

An agent produces a structured signal containing information such as:

```json
{
  "signal_type": "missing_results",
  "summary": "Potential missing-results signal...",
  "confidence": 0.92,
  "status": "pending"
}
```

Signals are persisted in PostgreSQL.

---

## 5. Uncertain Findings Enter Human Review

Signals that require additional validation can enter the review queue.

A reviewer can:

* approve
* reject
* edit

The review is persisted as an auditable record.

---

## 6. Human Feedback Becomes Memory

A rejection or correction can be written into procedural memory.

This allows future investigations to incorporate what the human reviewer previously taught the system.

```text
Agent Signal
     ↓
Human Review
     ↓
Reviewer Correction
     ↓
Procedural Memory
     ↓
Future Agent Reasoning
```

---

## 7. Final Intelligence Brief

The supervisor compiles the resulting signals into a final analysis response.

A production analysis response currently follows this general structure:

```json
{
  "run_id": "a5712262-7b94-4bfc-9d46-6861000681fe",
  "task": "Find completed clinical trials where results were never posted",
  "final_brief": "Three completed clinical trials identified...",
  "total_signals": 3,
  "signals_requiring_review": 0,
  "agents_activated": [
    "missing_results_agent",
    "track_record_agent"
  ],
  "duration_seconds": 15.24
}
```

---

# 🏗️ System Architecture

MOSAIC is organized into several layers:

```text
┌──────────────────────────────────────────────────────────────┐
│                     External Data Sources                    │
│                                                              │
│        ClinicalTrials.gov             PubMed                 │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                       Ingestion Layer                        │
│                                                              │
│  Clinical Trials Client │ PubMed Client │ Parser │ GCS       │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     Processing Layer                         │
│                                                              │
│       Chunking → Embedding Generation → Vector Storage       │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                       Memory Layer                           │
│                                                              │
│      Episodic Memory │ Procedural Memory │ Semantic Memory   │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Graph                     │
│                                                              │
│                       Supervisor                             │
│                            │                                 │
│        ┌───────────┬───────┼───────┬───────────┐             │
│        ▼           ▼       ▼       ▼           ▼             │
│   Missing      Broken   Track   Pattern     Side Effect      │
│   Results      Promises Record  Finder      Checker          │
│                                                        │     │
│                                             Timeline ──┘     │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     HITL Learning Layer                      │
│                                                              │
│           High Confidence       Low Confidence               │
│                  │                    │                       │
│                  ▼                    ▼                       │
│               Signals            Review Queue                │
│                                       │                      │
│                                       ▼                      │
│                              Procedural Memory               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                         FastAPI                              │
│                                                              │
│   Analysis │ Signals │ Review │ Memory │ Sponsors │ Health   │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
                  Google Cloud Run
```

---

# 📚 Data and Knowledge Pipeline

MOSAIC separates external data acquisition from knowledge processing.

## 1. Data Ingestion

### ClinicalTrials.gov

The ingestion layer retrieves clinical-trial records and normalizes them into application-level models.

### PubMed

The PubMed integration performs search and retrieval operations and parses research literature for downstream analysis.

### Google Cloud Storage

Raw and processed artifacts can be persisted in Google Cloud Storage.

```text
ClinicalTrials.gov ──┐
                     ├──> Ingestion ──> Parsing ──> GCS
PubMed ──────────────┘
```

The ingestion implementation includes retry handling and external API clients.

---

## 2. Document Processing

After ingestion, study information is converted into searchable chunks.

The current processing pipeline uses:

```text
Raw Document
     ↓
Chunking
     ↓
500-word chunks
50-word overlap
     ↓
Embedding Generation
     ↓
1536-dimensional vectors
     ↓
PostgreSQL + pgvector
```

---

## 3. Semantic Retrieval

When an agent needs supporting evidence:

```text
Agent Query
     ↓
Embedding
     ↓
Vector Similarity Search
     ↓
Relevant Chunks
     ↓
Study Context
     ↓
Agent Reasoning
```

This allows the agents to search by semantic meaning rather than only matching exact keywords.

---

# 🤖 Multi-Agent Reasoning

MOSAIC uses six specialist agents.

Each agent owns a distinct investigation responsibility.

| Agent                     | Responsibility                                                     |
| ------------------------- | ------------------------------------------------------------------ |
| **Missing Results Agent** | Identifies completed trials where results may not have been posted |
| **Broken Promises Agent** | Investigates potential outcome switching                           |
| **Track Record Agent**    | Builds and consults historical sponsor-level intelligence          |
| **Pattern Finder Agent**  | Looks for patterns across multiple studies                         |
| **Side Effect Checker**   | Investigates potential safety discrepancies across sources         |
| **Timeline Analyst**      | Identifies unexplained delays relative to study timelines          |

---

## Why Specialist Agents?

A single general-purpose agent would have to simultaneously reason about:

* result-posting behaviour
* study timelines
* sponsor history
* outcome changes
* cross-study patterns
* safety evidence

MOSAIC instead separates these responsibilities.

```text
                    Supervisor
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
 Missing Results   Broken Promises   Track Record
       │                │                │
       └─────────┬──────┴──────┬─────────┘
                 │             │
                 ▼             ▼
          Pattern Finder   Side Effects
                 │
                 ▼
             Timeline
```

---

## Parallel Execution

The specialist agents are designed to run independently.

```text
                 Supervisor
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
     Agent A      Agent B       Agent C
        │            │            │
        ▼            ▼            ▼
     Result       Result        Result
        │            │            │
        └────────────┼────────────┘
                     ▼
                 Aggregation
```

This reduces unnecessary sequential dependency between independent investigations.

---

## Agent Confidence

Each specialist can use an agent-specific confidence threshold before a finding proceeds.

Current thresholds include:

| Agent               | Threshold |
| ------------------- | --------: |
| Missing Results     |      0.65 |
| Broken Promises     |      0.60 |
| Track Record        |      0.70 |
| Pattern Finder      |      0.65 |
| Side Effect Checker |      0.55 |
| Timeline Analyst    |      0.60 |

These thresholds are part of the system's signal-routing logic rather than representing a universal clinical or regulatory standard.

---

# 🧠 Memory Architecture

MOSAIC maintains three distinct forms of persistent memory.

```text
                    MOSAIC MEMORY
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
     Episodic        Procedural        Semantic
        │                │                │
        ▼                ▼                ▼
   Past sessions     Learned rules    Sponsor knowledge
```

## Episodic Memory

**Question answered:**

> What happened during previous analysis sessions?

Stores information about previous investigations and findings.

---

## Procedural Memory

**Question answered:**

> What has human feedback taught the agents?

Stores reasoning rules and corrections derived from human review.

Example:

```text
Agent generates signal
        ↓
Human rejects signal
        ↓
Rejection reason recorded
        ↓
Procedural memory updated
        ↓
Future reasoning incorporates the correction
```

---

## Semantic Memory

**Question answered:**

> What accumulated knowledge do we have about a sponsor?

Sponsor profiles accumulate information such as:

* total studies
* results posted
* average delay
* broken promises
* credibility score

This allows sponsor-level context to persist across independent analysis sessions.

---

# 👤 Human-in-the-Loop Learning

MOSAIC does not treat every generated signal as automatically correct.

The review pipeline is:

```text
                Agent Signal
                     │
                     ▼
              Confidence Check
                     │
             ┌───────┴───────┐
             ▼               ▼
      High Confidence    Low Confidence
             │               │
             ▼               ▼
       Persist Signal    Review Queue
                             │
                             ▼
                       Human Reviewer
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
                 Approve   Reject    Edit
                             │
                             ▼
                    Procedural Memory
```

Each review decision is stored for auditability.

A review contains information such as:

```text
Signal ID
Reviewer
Decision
Edit Summary
Rejection Reason
Timestamp
```

This creates two benefits:

### 1. Auditability

The system can retain a record of what was reviewed and how the decision was made.

### 2. Learning

Reviewer corrections can become procedural memory that influences future reasoning.

> **Important:** this learning loop is persistent reasoning-memory adaptation; it is not model fine-tuning.

---

# 🗄️ Database Design

MOSAIC uses PostgreSQL as its primary relational persistence layer, with pgvector for semantic retrieval.

The core schema consists of five tables:

```text
                         ┌───────────┐
                         │  studies  │
                         └─────┬─────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
               ┌─────────┐          ┌─────────┐
               │ chunks  │          │ signals │
               └─────────┘          └────┬────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │ hitl_reviews│
                                   └─────────────┘

                         ┌──────────────────┐
                         │ sponsor_profiles │
                         └──────────────────┘
```

## `studies`

Canonical record for each clinical trial.

Important fields include:

```text
nct_id
title
sponsor
phase
status
conditions
interventions
primary_outcome
secondary_outcomes
start_date
completion_date
results_posted
enrollment
```

Every other study-related entity can reference the study using `nct_id`.

---

## `chunks`

Stores processed document chunks together with their vector embeddings.

```text
chunk_id
nct_id
chunk_text
embedding
chunk_index
source
```

The embedding column is used for semantic similarity search through pgvector.

---

## `signals`

Stores findings generated by agents.

```text
signal_id
nct_id
agent
signal_type
summary
evidence
confidence
status
created_at
```

Possible signal states include:

```text
pending
approved
rejected
edited
```

The `signals` table represents the persistent output of the investigation layer.

---

## `hitl_reviews`

Stores human review decisions.

```text
review_id
signal_id
reviewer
decision
edit_summary
rejection_reason
reviewed_at
```

This provides the audit trail required for the human-review workflow and supplies information to procedural memory.

---

## `sponsor_profiles`

Stores accumulated sponsor-level information.

```text
sponsor
credibility_score
total_studies
results_posted
avg_delay_days
broken_promises
last_updated
```

The profile is updated as additional studies are analyzed.

The current credibility calculation uses:

```text
70% → results compliance rate
30% → promise-keeping component
```

## with the resulting score represented on a 0–1 scale.

## Database Flow

```text
ClinicalTrials.gov
       │
       ▼
    studies
       │
       ▼
    chunks
       │
       ▼
 Vector Search
       │
       ▼
    Agents
       │
       ▼
   signals
       │
       ▼
 Human Review
       │
       ▼
 hitl_reviews
       │
       ▼
Procedural Memory

                    ┌─────────────────┐
                    │ sponsor_profiles│
                    └────────▲────────┘
                             │
                       Track Record
```

---

# 📡 API Architecture

MOSAIC exposes its functionality through FastAPI.

## Analysis

```http
POST /api/v1/analyze
```

Triggers an analysis run.

## Signals

```http
GET /api/v1/signals
GET /api/v1/signals/{id}
```

Retrieves generated signals.

## Human Review

```http
GET   /api/v1/review/queue
PATCH /api/v1/review/{id}
```

Retrieves pending reviews and records reviewer decisions.

## Memory

```http
GET /api/v1/memory/episodes
GET /api/v1/memory/procedures/{agent}
```

Provides access to episodic and procedural memory.

## Sponsor Intelligence

```http
GET /api/v1/sponsors
GET /api/v1/sponsors/{name}
```

Retrieves accumulated sponsor-level intelligence.

## Health

```http
GET /api/v1/health
```

Provides a system health check.

Interactive API documentation is automatically available through FastAPI's Swagger interface at:

```text
/docs
```

---

# 🧩 Key Engineering Decisions

## 1. Why a Multi-Agent Architecture?

### Requirement

Different research-integrity questions require different investigation logic.

### Decision

Use specialist agents with explicit responsibilities.

### Benefit

Each investigation can be developed, tested, and reasoned about independently.

### Trade-off

More components and orchestration complexity compared with a single-agent implementation.

---

## 2. Why Parallel Agent Execution?

Independent investigations do not always need to wait for one another.

MOSAIC therefore uses graph-based orchestration to allow specialist agents to execute concurrently.

### Trade-off

Parallel execution introduces aggregation and state-management complexity.

---

## 3. Why PostgreSQL + pgvector?

The system requires both:

* structured relational data
* semantic retrieval

Using PostgreSQL with pgvector allows both forms of persistence to coexist in the same database layer.

The `chunks` table stores text alongside vector embeddings and supports vector similarity retrieval.

---

## 4. Why Three Memory Types?

Different kinds of knowledge have different lifecycles.

```text
Episodic
"What happened?"

Procedural
"What did humans teach us?"

Semantic
"What do we know?"
```

Separating them makes the memory model easier to reason about and maintain.

---

## 5. Why Human-in-the-Loop?

Research-integrity signals can require contextual interpretation.

Rather than automatically treating every model output as correct, uncertain findings can be routed to human review.

This also creates a feedback mechanism for improving future agent reasoning.

---

## 6. Why Cloud Run?

The API is deployed using Google Cloud Run.

This provides a serverless container execution model and allows the service to scale down when idle.

The supporting infrastructure includes Cloud SQL, Cloud Storage, and Secret Manager.

---

# 🧪 Challenges and Solutions

## Challenge 1 — Coordinating Multiple Investigations

### Problem

Different agents need access to common state while maintaining their own responsibilities.

### Approach

Use a shared graph state and explicit LangGraph orchestration.

```text
Supervisor
    ↓
Shared State
    ↓
Parallel Agents
    ↓
Aggregated Results
```

---

## Challenge 2 — Combining Structured and Semantic Retrieval

### Problem

Some investigations depend on exact structured fields, while others require semantic understanding of study text.

### Approach

Use PostgreSQL for structured records and pgvector for semantic retrieval.

This allows the agents to combine database queries with evidence retrieved from embedded document chunks.

---

## Challenge 3 — Handling Uncertain Signals

### Problem

A generated signal may not contain enough context to be treated as reliable automatically.

### Approach

Attach confidence to generated signals and route cases requiring additional validation through the HITL review queue.

---

## Challenge 4 — Making Human Feedback Persistent

### Problem

A correction is not very useful if it disappears after a single session.

### Approach

Persist review decisions and rejection reasons and expose them through procedural memory.

```text
Human Correction
       ↓
Persistent Storage
       ↓
Procedural Memory
       ↓
Future Agent Runs
```

---

## Challenge 5 — External API Reliability

The ingestion layer interacts with external research APIs.

The implementation therefore separates external clients from downstream processing and includes retry handling for transient failures.

---

# 📊 Evaluation

MOSAIC includes a dedicated evaluation package:

```text
evaluation/
├── eval_runner.py
└── scorers.py
```

The evaluation layer is intended to provide a repeatable mechanism for assessing agent outputs and system behaviour.

Current evaluation work focuses on building the evaluation infrastructure and expanding quantitative benchmarks.

Future evaluation work includes automated LLM evaluation and broader signal-level benchmarking.

---

# ☁️ Deployment

MOSAIC is designed for deployment on Google Cloud Platform.

## Production Architecture

```text
                       Internet
                          │
                          ▼
                   Google Cloud Run
                          │
                          ▼
                      FastAPI
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
      Cloud SQL       Cloud Storage     Secret Manager
          │
          ▼
 PostgreSQL + pgvector
          │
          ▼
      MOSAIC Data
```

External services:

```text
ClinicalTrials.gov
PubMed
OpenAI API
```

---

## Deploy

```bash
chmod +x deployment/gcp/deploy.sh

./deployment/gcp/deploy.sh
```

The deployment configuration uses an AMD64 container for Cloud Run.

Manual deployment:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag gcr.io/YOUR_PROJECT_ID/mosaic-api:latest \
  --file deployment/Dockerfile \
  --push .
```

Then:

```bash
gcloud run deploy mosaic-api \
  --image=gcr.io/YOUR_PROJECT_ID/mosaic-api:latest \
  --platform=managed \
  --region=us-central1 \
  --memory=2Gi \
  --cpu=2 \
  --port=8000 \
  --project=YOUR_PROJECT_ID
```

---

## Infrastructure Cost

The current project documentation estimates the following approximate running costs:

| Service       | Estimated Monthly Cost |
| ------------- | ---------------------: |
| Cloud Run     |                  ~$3–8 |
| Cloud SQL     |                ~$15–18 |
| Cloud Storage |                 ~$0.03 |
| OpenAI API    |                ~$15–25 |
| **Total**     |      **~$35–55/month** |

Idle cost can be reduced by stopping Cloud SQL when it is not required. These figures are deployment estimates rather than guaranteed billing amounts.

---

# 📁 Project Structure

```text
mosaic/
│
├── agents/
│   ├── supervisor.py
│   ├── broken_promises_agent.py
│   ├── missing_results_agent.py
│   ├── track_record_agent.py
│   ├── pattern_finder_agent.py
│   ├── side_effect_agent.py
│   └── timeline_agent.py
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── dependencies.py
│   └── routers/
│       ├── analysis.py
│       ├── memory.py
│       ├── review.py
│       └── signals.py
│
├── config/
│   ├── settings.py
│   └── logging_config.py
│
├── graph/
│   ├── state.py
│   ├── graph_builder.py
│   └── hitl.py
│
├── ingestion/
│   ├── clinical_trials_client.py
│   ├── pubmed_client.py
│   ├── document_parser.py
│   └── gcs_store.py
│
├── processing/
│   ├── chunker.py
│   ├── embedder.py
│   └── vector_store.py
│
├── memory/
│   ├── episodic_store.py
│   ├── procedural_store.py
│   └── semantic_store.py
│
├── tools/
│   ├── clinical_tools.py
│   ├── pubmed_tools.py
│   └── search_tools.py
│
├── evaluation/
│   ├── eval_runner.py
│   └── scorers.py
│
├── deployment/
│   ├── Dockerfile
│   └── gcp/
│       ├── cloud_run.yaml
│       └── cloudsql_init.sql
│
├── tests/
│   ├── test_agents.py
│   ├── test_api.py
│   ├── test_ingestion.py
│   ├── test_memory.py
│   └── test_processing.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── .env.example
├── requirements.txt
├── README.md
└── .gitignore
```

## Where to Start

If you want to understand the core system quickly, start with:

```text
1. graph/graph_builder.py
        ↓
2. agents/supervisor.py
        ↓
3. agents/missing_results_agent.py
        ↓
4. graph/hitl.py
        ↓
5. api/routers/analysis.py
```

These files provide a useful path through orchestration → agent reasoning → human review → API exposure.

---

# 🚀 Getting Started

## Prerequisites

* Python 3.12+
* Docker Desktop
* Google Cloud account
* OpenAI API key
* PostgreSQL / Cloud SQL access

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/mosaic.git

cd mosaic
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt

pip install -e .
```

---

## 4. Configure Environment

```bash
cp .env.example .env
```

Then configure the required values.

```env
# OpenAI
OPENAI_API_KEY=your-api-key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o

# GCP
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=your-bucket-name

# Cloud SQL
DB_HOST=your-database-host
DB_PORT=5432
DB_NAME=clinical_trial_db
DB_USER=mosaic_user
DB_PASSWORD=your-password

# API
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development
```

> **Never commit `.env` to Git.**

---

# 🗃️ Database Setup

Create the database and execute:

```text
deployment/gcp/cloudsql_init.sql
```

The schema creates:

```text
studies
chunks
signals
hitl_reviews
sponsor_profiles
```

The database also configures vector indexing for semantic retrieval.

---

# 📥 Run the Ingestion Pipeline

```bash
python3 ingestion/run_ingestion.py
```

This downloads and processes study/literature data from the configured external sources and stores raw/processed artifacts.

---

# ⚙️ Run the Processing Pipeline

```bash
python3 processing/run_processing.py
```

This performs:

```text
Documents
   ↓
Chunking
   ↓
Embedding generation
   ↓
Vector persistence
```

---

# 🚀 Run the API

```bash
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload
```

Open:

```text
http://localhost:8000/docs
```

to access the interactive Swagger UI.

---

# 📡 API Usage

## Run an Analysis

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Find completed trials with missing results",
    "max_studies": 5
  }' \
  http://localhost:8000/api/v1/analyze | python3 -m json.tool
```

---

## Check the Review Queue

```bash
curl -s \
  http://localhost:8000/api/v1/review/queue | python3 -m json.tool
```

---

## Submit a Review

```bash
curl -s -X PATCH \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "reject",
    "reviewer": "analyst@company.com",
    "rejection_reason": "Trial was terminated early and therefore requires different interpretation."
  }' \
  http://localhost:8000/api/v1/review/QUEUE_ID_HERE | python3 -m json.tool
```

---

## Search Episodic Memory

```bash
curl -s \
  "http://localhost:8000/api/v1/memory/episodes?query=missing+results+sponsor" \
  | python3 -m json.tool
```

---

## Inspect Agent Procedures

```bash
curl -s \
  http://localhost:8000/api/v1/memory/procedures/missing_results_agent \
  | python3 -m json.tool
```

---

# 🧪 Testing

Run the complete test suite:

```bash
pytest
```

Tests are organized by system layer:

```text
tests/
├── test_agents.py
├── test_api.py
├── test_ingestion.py
├── test_memory.py
└── test_processing.py
```

The test suite covers:

* agent behaviour
* API endpoints
* ingestion functionality
* memory operations
* processing and retrieval components

---

# ⚠️ Limitations and Future Work

MOSAIC is currently focused on demonstrating the architecture for multi-agent clinical-trial intelligence.

## Current Limitations

* Evaluation coverage is still being expanded.
* The primary external research sources are currently ClinicalTrials.gov and PubMed.
* Sponsor-level intelligence becomes more useful as historical study data accumulates.
* A dedicated visual review dashboard is not yet part of the current API-first implementation.
* Automated evaluation of generated signals is still being expanded.

## Future Work

* [ ] Add automated evaluation with LangSmith
* [ ] Build a dedicated Streamlit signal-review dashboard
* [ ] Integrate FDA adverse-event data
* [ ] Add scheduled analysis using Cloud Scheduler
* [ ] Add email alerts for high-confidence signals
* [ ] Build sponsor comparison functionality
* [ ] Add international trial registries such as EudraCT and ISRCTN
* [ ] Evaluate domain-specific embedding models

---

# 👨‍💻 My Contributions

I designed and implemented the core MOSAIC system across the application and infrastructure layers, including:

* Multi-agent architecture and LangGraph orchestration
* Six specialist investigation agents
* Supervisor-based agent coordination
* Parallel investigation workflow
* Episodic, procedural, and semantic memory
* Human-in-the-loop review and feedback flow
* ClinicalTrials.gov and PubMed ingestion
* Document processing and chunking
* Embedding generation and vector retrieval
* PostgreSQL + pgvector persistence
* Sponsor intelligence storage
* FastAPI REST API
* Agent tools and search interfaces
* Automated tests
* Docker-based deployment
* Google Cloud deployment configuration

---

# 📌 Architecture Summary

The core idea behind MOSAIC can be summarized in one flow:

```text
                    DATA
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
ClinicalTrials.gov              PubMed
        │                         │
        └────────────┬────────────┘
                     ▼
                INGESTION
                     │
                     ▼
               PROCESSING
              Chunk + Embed
                     │
                     ▼
             KNOWLEDGE STORE
           PostgreSQL + pgvector
                     │
                     ▼
                  MEMORY
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Episodic     Procedural   Semantic
        │            │            │
        └────────────┼────────────┘
                     ▼
                 SUPERVISOR
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
   Specialist    Specialist    Specialist
    Agents        Agents        Agents
       │             │             │
       └─────────────┼─────────────┘
                     ▼
                  SIGNALS
                     │
                     ▼
                  HITL
                     │
              ┌──────┴──────┐
              ▼             ▼
           Accept         Review
                            │
                            ▼
                   Procedural Memory
                            │
                            ▼
                    Future Reasoning
```

The result is an AI system that combines **data ingestion, semantic retrieval, multi-agent reasoning, persistent memory, human oversight, and production infrastructure** into one end-to-end architecture.

---
