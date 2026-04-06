# AtomicX Architecture — The Developer’s Deep-Dive 🚀

AtomicX is a **Universal Causal Intelligence Engine**. It is designed to move beyond traditional "probabilistic" trading bots toward a **Sovereign Predictive Intelligence** that reasons through reality using a 7-layer cognitive hierarchy.

---

## 1. The Core Technology Stack

- **Language**: **Python 3.13 (NoGIL)** — Optimized for extreme concurrency in multi-agent swarms.
- **Storage**: **PostgreSQL + TimescaleDB** — Specialized for high-frequency time-series data (OHLCV, Ticks, Liquidations).
- **Processing**: **Polars** — Blazing fast DataFrame engine for calculating 100+ indicators in <10ms.
- **AI/LLM**: **DeepSeek-R1 & Claude 3.5 Sonnet** — Orchestrated for "System 2" slow-thinking and causality mapping.
- **Communication**: **Apache Kafka** — For asynchronous event-driven signals across node layers.

---

## 2. The 7-Layer Hierarchy (Logical Flow)

The system operates in a **Perception → Reason → Evolution** loop:

### **L1: Data Ingestion (The Senses)**
- **File**: `src/atomicx/data/connectors/binance.py`
- **What it does**: Establishes raw WebSocket connections. It ingests OHLCV (price), Ticks, Order Books, and the **Pain Map** (Liquidations).

### **L2: Variable Engine (The Features)**
- **File**: `src/atomicx/variables/engine.py`
- **What it does**: Computes 46+ causal variables (RSI, ADX, CVD, VWAP) in real-time using vectorized Polars math. These are fed to the agents as "The Truth."

### **L4: Debate Chamber (The Brain)**
- **File**: `src/atomicx/brain/debate.py`
- **What it does**: This is where the **Intelligence** happens. It initiates a "Consensus Protocol" between multiple specialized agents (e.g., `WhaleAgent`, `PatternAgent`).
- **The Upgrade**: It uses **Reasoning Traces (DeepSeek-R1 style)**. Agents perform internal `<thought>` deliberation to map out *why* a move is occurring before giving a final prediction.

### **L7: Narrative Layer (The Context)**
- **File**: `src/atomicx/narrative/__init__.py`
- **What it does**: Scans Twitter/X, News, and Reddit to detect **Narrative Clusters**. It identifies if a price move is driven by "Fundamentals" or just "Sentiment Noise."

### **L12: Fusion & Confidence (The Gatekeeper)**
- **File**: `src/atomicx/fusion/__init__.py`
- **What it does**: Combines the Mathematical signals (L2) with the Causal Reasoning (L4) and Narrative Context (L7). It only triggers a trade if the **Confidence Gate** (0.65+) is surpassed.

### **L14: Evolution Engine (The Self-Correction)**
- **File**: `src/atomicx/evolution/self_improvement.py`
- **What it does**: This is a Reinforcement Learning loop. It "Grades" the bot’s reasoning traces against reality. If the bot was "Right for the Wrong Reasons," it penalizes the agent's trust score.

---

## 3. Advanced Memory: The "Dreaming" Loop

AtomicX doesn't just "store" data; it **compounds it**.

- **Vector Memory (Qdrant)**: Stores fast, episodic memories of recent trade outcomes.
- **Obsidian Wiki (Knowledge Base)**: The system writes **Markdown Research Notes** (via `obsidian-wiki` pattern) to your local filesystem.
- **The "Dreaming" Phase**: Every 24 hours, the system asynchronously consolidates its "Daily Notes" into "Causal Laws." It’s the first trading system that literally **writes its own textbooks.**

---

## 4. The Singularity Tier: Self-Healing & URM

- **The Medic (ASH)**: A dedicated agent that monitors for 403 blocks, API failures, or logic crashes. It **takes action** (rotates proxies, switches APIs) without manual developer input.
- **Universal Reality Modeling (URM)**: The ability to predict non-financial events (Geopolitics) by treating Leaders and Countries as "Actors" with "Causal Variables."

---

## 5. Summary for the Developer

If you are looking at the code, focus on the **`CognitiveLoop`**. It is the orchestrator that pulls data from **L1**, feeds it to **L4**, and saves the "Lessons" in **L14**. 

AtomicX is built to be **Sovereign**—it manages its own infrastructure, audits its own logic, and learns from its own history.
