# AI Engineering & Machine Learning Roadmap

---

## Foundations: Deep Learning & LLMs from Scratch
* **TensorTonic** — *Language Modelling from Scratch*
* **Jay Alammar** — *The Illustrated Transformer*
* **Stanford CS336** — *Language Modeling from Scratch*
* **Stanford CS329A** — *Self-Improving AI Agents / Machine Learning Systems*
* **Andrej Karpathy** — *The Spelled-Out Intro to Neural Networks and Backpropagation*
* **TechSimPlus Learning** — *Gen AI with Python and LangChain*

---

## Core Machine Learning Projects (Must-Build)

### 1. Demand Forecasting
* **Goal:** Predict how much a product will sell.
* **Key Tasks:** Explore sales patterns over time and use past sales data to forecast future demand.

### 2. Customer Churn Predictor
* **Goal:** Predict which customers are likely to leave.
* **Key Tasks:** Explore patterns in customer activity; compare Logistic Regression with a Decision Tree.

### 3. NBA Match Predictor
* **Goal:** Predict which team will win.
* **Key Tasks:** Create features from past games and team performance; test predictions on more recent matches.
* **Core Takeaway:** *Feature engineering* teaches you how to turn raw statistics into useful signals.

### 4. NLP Insights
* **Goal:** Find patterns in customer reviews.
* **Key Tasks:** Convert review text into numerical features; classify sentiment and inspect incorrect predictions.
* **Core Takeaway:** Working with text teaches you how models learn from language and where they struggle.

### 5. Production ML: Fraud Detector
* **Goal:** Build a fraud detector you can trust.
* **Key Tasks:** Create a reproducible pipeline for preprocessing and training; check for data leakage and evaluate missed fraud cases vs. false alarms.
* **Core Takeaway:** *Production-grade ML* means making the process reliable from raw data to predictions.

---

## Agentic AI Courses
1. 🤗 **Hugging Face** — *Agentic AI Course*
2. 🤝 **DeepLearning.AI** — *Multi AI Agent Systems with CrewAI*
3. 🦜 **LangChain** — *Introduction to LangGraph with Python*
4. 🧠 **Andrew Ng + DeepLearning.AI** — *Agentic AI*
5. 🎓 **UC Berkeley** — *Advanced Large Language Model Agents*

---

## Agentic AI in a Week (Video & Crash-Course Playlist)
* **AI Agents Clearly Explained** — *Jeff Su*
* **What are AI Agents?** — *IBM Technology*
* **Building AI Agents in Pure Python** — *Dave Ebbelaar*
* **LangGraph Complete Course** — *freeCodeCamp*
* **Learn RAG from Scratch** — *freeCodeCamp*
* **Tips for Building AI Agents** — *Anthropic*
* **Complete Agentic AI Course in 10 Hours** — *Krish Naik*

---

## AI Engineering Talks & Best Practices
* **12-Factor Agents: Patterns of Reliable LLM Applications** — *AI Engineer*
* **No Vibes Allowed: Solving Hard Problems in Complex Codebases** — *AI Engineer*
* **How We Build Effective Agents** — *AI Engineer*
* **Don't Build Agents, Build Skills Instead** — *AI Engineer*
* **Building and Evaluating AI Agents** — *AI Engineer*
* **Beating RL with Reflections** — *AI Engineer*
* **Carrying Out Error Analysis** — *DeepLearning.AI*

---

## Prompting Tricks & Workflows
* **ASD-STE100 (Simplified Technical English):** Use constrained, unambiguous phrasing (gets you 80% of the way to reliable outputs).
* **Visual over Text:** Ask for diagrams (e.g., Mermaid, ASCII, SVG) instead of walls of text.
* **Interactive Outputs:** Ask for standalone HTML/JS prototypes instead of static explanations.
* **Explainer Videos:** Prompt for code/scripts to generate *3Blue1Brown*-style animations (e.g., via Manim).

---

## Production RAG Projects

### 1. Semantic Caching & Cost-Aware Routing
* **What to Build:** Cache answers by query embedding similarity, route simple questions to a small model and hard ones to a large model, and track cost and latency per request on a dashboard.
* **Why It Matters:** Decides whether a RAG product is affordable at scale.
* > **Resume Bullet:** *"Added semantic caching and model routing to a RAG service, reducing LLM cost and p95 latency while maintaining answer quality."*

### 2. Permission-Aware Multi-Tenant RAG
* **What to Build:** Attach access metadata to every chunk, filter retrieval by the user's role *before* context reaches the LLM, and maintain an audit log recording what was retrieved for whom.
* **Why It Matters:** Solves the primary security blocker when RAG touches internal company data.
* > **Resume Bullet:** *"Built a multi-tenant RAG service with role-based retrieval filtering and audit logging, preventing cross-tenant data leakage by design."*

### 3. GraphRAG for Multi-Hop Questions
* **What to Build:** Extract entities and relationships from documents into a knowledge graph, then combine graph traversal with vector search.
* **Why It Matters:** Plain RAG fails when the answer is spread across multiple documents; graph traversal connects them.
* > **Resume Bullet:** *"Built a GraphRAG system combining knowledge graph traversal with vector retrieval, improving accuracy on multi-hop questions over a standard RAG baseline."*

---

## Agentic AI & Applied Projects

### Core AI Engineering Agents
1. **Structured Output Agent**
   * Enforce Pydantic JSON schemas, validate tool responses, retry on parse errors, and log validation failures.
2. **RAG Agent with Citation Grounding**
   * Retrieve context, generate answers with inline sources, flag low-confidence responses, and fallback to web search.
   * *Proves:* You can prevent hallucinations at scale.
3. **ReAct Planning Agent**
   * Implement an *Observe → Think → Act → Reflect* loop with max iteration limits, self-critique, and graceful degradation.
   * *Proves:* You can build agents that don't get stuck in infinite loops.
4. **Multi-Tool Orchestrator Agent**
   * Build a dynamic tool registry with capability-based routing, permission scoping, parallel execution, and conflict resolution.
   * *Proves:* You can manage complex multi-tool coordination cleanly.
5. **Memory-Enabled Conversational Agent**
   * Combine a short-term buffer with long-term vector recall, context compression, relevance scoring, and cross-session sync.
   * *Proves:* You can build stateful agents that genuinely remember users.

### Role-Specific & App Builds
* **FDE (Forward Deployed Engineer) Project:** Customer Support Resolution Agent (end-to-end ticket triage, tool execution, and escalation).
* **Full-Stack GenAI App:** Custom YouTube NotebookLM (transcript ingestion, timestamped citations, Q&A, and audio/study-guide synthesis).
