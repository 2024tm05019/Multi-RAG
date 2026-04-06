# Multi-RAG
Containing Multimodal RAG 
1.	**Domain:** Automobile Weld Shop project Execution & Maintenance Troubleshooting (Engineering) 
2.	**Problem Description:** Automotive Weldshop Contains many unique Equipment sourced from different OEMs.e.g. Robots, Robot Controllers, Weld Controllers,Manual Sealer System,co-ordination Measurement Machine,FARO portable measurement machine,Hand Manipulators,Grippers,Mylers,Shims,Weldments,Fixture Baseplate,Tooling Holes,DC Nut Runner,Battery Gun,Battery Gun Sockets,Wifi DC Nut Runner,Bluetooth based DC Nut Runner,KBK Rails,Turck Communication Modules,Allen Bradley Programmable Logic Controller,Safety Programmable Logic Controllers,Limit Switches,Inductive Sensors,Pneumatic Cylinders,Laser Sensors,Dump Cylinders, Air Balancer,Electric Hoist,Chain Conveyors,Powered Roller Bed,Skid,Geo Pallet, Auto Sealer systems,Auto Weld Guns, Manual Weld Guns,Auto Tip Dresser, Manual Tip Dressers, Spot Welding System,,Spot welding Fixed Arm,Spot Welding Movable Arm,Welding Transformers,Cooling Circuit,Air Circuit,Air Compressors,Water Cooling Towers,Chillers,Fume Extractors,Laser Welding System,Projection Welding System,Piercing Sysem,Laser Brazing,Co2 Welding,Mig Welding,Tig Welding,Jig,Fixtures,Optimum usage formulas and Troubleshooting Codes with error descriptions.
3.	**Why This Problem Is Unique:** Troubleshooting during the Shutdown execution and Breakdown of the equipment during Peak production always have a time crunch. People are mostly depends on specific machine or equipment supplier's recommendation or experts opinion.Project execution or production are happening round the clock while expert opinion may not be available for 24x7. (e.g., specialised terminology, regulatory tables, engineering diagrams,charts with fine-print footnotes).
4.	**Why RAG Is the Right Approach:** By using this Production grade RAG Maintenance and Project team can get expert like opinions through questioning specific questions. Each equipment have specific unique concept and design which RAG can describe uniquely rather than general answers from LLM 
5.	**Expected Outcomes:** Successful System will enable expert like opinions to the questions which will solve Breakdowns quickly.

## Architecture Overview
graph TD
    A[PDF Upload - Bosch Manual] --> B[PyMuPDF Parser]
    B --> C[Text Chunks]
    B --> D[Table Chunks]
    B --> E[Image Extraction]
    E --> F[GPT-4o Vision - Image Summarization]
    C --> G[OpenAI Embeddings]
    D --> G
    F --> G
    G --> H[ChromaDB Vector Store]
    I[User Query] --> J[OpenAI Embeddings]
    J --> K[ChromaDB Retriever Top-5]
    H --> K
    K --> L[GPT-4o LLM + Custom Prompt]
    L --> M[Answer + Source References]
## Technology Choices

| Component | Choice | Justification |
|---|---|---|
| PDF Parser | PyMuPDF (fitz) | Best support for extracting images, text blocks, and page metadata from complex industrial PDFs |
| Embedding Model | OpenAI text-embedding-3-small | High accuracy, low cost, 1536-dim vectors suitable for technical terminology |
| Vector Store | ChromaDB | Supports metadata filtering by chunk_type (text/table/image) — essential for cross-modal retrieval |
| LLM | GPT-4o | Strong technical reasoning; can handle industrial domain terminology |
| Vision Model | GPT-4o | Multimodal — can summarize circuit diagrams and connector pinouts accurately |
| Framework | Custom (FastAPI + direct ChromaDB) | More control over chunk metadata and retrieval logic than LangChain abstractions |
