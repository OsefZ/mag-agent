# Mnemonic Addressable Graphs for LLM Agents (MAG)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

This repository contains the official implementation and experimental data for the MSc thesis, **"Mnemonic Addressable Graphs for Token-Efficient Code-Generation Agents"**.

MAG is a novel dynamic memory architecture for Large Language Model (LLM) agents that enables complex, multi-step reasoning. It moves beyond traditional vector-based retrieval by recording an agent's actions and their outcomes as a queryable, causal graph. This allows the agent to reflect on its own history, learn from its mistakes, and perform robust self-correction.

---

## Core Features

* **Dynamic Graph Memory:** The agent's experiences (e.g., generating a `Patch`, observing a `TestFailure`) are recorded as structured nodes and edges in a property graph.
* **Symbolic Retrieval:** Agents retrieve context by issuing precise, symbolic queries (e.g., "find my last patch that failed and the corresponding error log"), eliminating the ambiguity of semantic search.
* **Structured Reflection & Self-Correction:** By querying its own history, the MAG agent can understand the consequences of its actions, enabling a powerful self-correction loop that dramatically improves success rates on complex tasks.
* **Token Efficiency:** By retrieving minimal, highly-relevant context, MAG reduces prompt sizes and makes multi-step agentic workflows economically feasible.

---

## Setup Instructions

### Prerequisites

-   Python 3.9+
-   `pip` and `git`
-   **Docker** (for running a local graph database)


```bash
conda create --name msc_project python=3.9
conda activate msc_project
cp .env.example .env

conda install -c anaconda cmake
pip install -r requirements.txt
```

### 1. Deploy Memgraph Database

The MAG Agent requires **Memgraph**, a high-performance graph database for storing code relationships and dependencies. Deploy it with Docker:

```bash
# Deploy Memgraph with web interface
docker run -d --name memgraph-mag \
  -p 7687:7687 -p 7444:7444 -p 3000:3000 \
  memgraph/memgraph-platform

# Verify deployment
docker ps | grep memgraph
```

**Database Access:**
- **Bolt Protocol**: `bolt://localhost:7687` (for agent connections)
- **Memgraph Lab**: `http://localhost:7444` (visual graph explorer)
- **Monitoring**: `http://localhost:3000` (performance dashboard)

**Production Features:**
- Sub-second query performance for 1000+ nodes
- Automatic relationship inference and caching
- ACID compliance with crash recovery

### 2. Clone and Install the Package

Clone the repository and install it in editable mode. This allows you to modify the code and run it without reinstalling.

```sh
git clone [https://github.com/Henry8772/mag-agent.git](https://github.com/Henry8772/mag-agent.git)
cd mag-agent
pip install -r requirements.txt
pip install -e .
```
