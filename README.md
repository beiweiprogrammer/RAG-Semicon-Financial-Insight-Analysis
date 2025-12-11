# Semicon-Insight: Intelligent Financial RAG Analyst 📈
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.10+-purple.svg)
![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)
![Status](https://img.shields.io/badge/Status-Prototype-green.svg)
> **An Enterprise level mulitomodal RAG system, designed for Semiconductor industry(NVIDIA VS AMD) 10-k financial report deep analysis**
## 📖  Background
Traditional rag system would face two difficulties when dealing with enterprise 10k financial report.
1. **Complex tables analysis losses efficacy** Cross-page tables in pdf might be interpreted as garbled characters, preventing LLM from answering specific financial numerical questions.
2. **Lack macroscopic comparison ability** Simple vector search cannot solve "Compare the two company's strategies" such macropic problems.
** Semicon-Insight**: By introducing Multimodal-Analysis, Hybrid-Chuncking and Router-Chucking, I realize efficient  unstructured text and structured financial data.
## System Architecture
The system utilized **Router-based Agentic RAG** structure, pick up search strategy based on user intention.
```mermaid
graph TD
    User[User Query] --> Router{Router Query Engine}
    
    subgraph "Data Ingestion Layer"
        PDF[10-K PDFs] -->|LlamaParse| MD[Markdown]
        MD -->|MarkdownElementNodeParser| Nodes[Text & Table Nodes]
    end

    subgraph "Indexing Layer"
        Nodes -->|Embedding| Qdrant[(Unified Qdrant Vector DB)]
        Nodes -->|Summary| SumNVDA[NVIDIA Summary Index]
        Nodes -->|Summary| SumAMD[AMD Summary Index]
    end

    Router -- "Specific Fact / Comparison" --> ToolA[Vector Search Tool]
    Router -- "NVIDIA Overview" --> ToolB[NVDA Summary Tool]
    Router -- "AMD Overview" --> ToolC[AMD Summary Tool]

    ToolA -->|Hybrid Search + Re-ranking| LLM
    ToolB -->|Tree Summarize| LLM
    ToolC -->|Tree Summarize| LLM

    LLM --> Response

