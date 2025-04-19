# Deepsearch Repo Walkthrough

This document provides an overview of the Deepsearch repository, its main components, and the workflow of the multi-agent system using an arrow and boxes schema for clarity.

---

## Repo Overview

- A replica of OpenAI Deepsearch.
- Implements a multi-agent system for deep research queries.
- Uses OpenAI GPT models via LangChain.
- Supports terminal and Gradio UI interfaces.
- Agents: Planning, Reasoning, Confirmation, Execution.
- Prompts define agent behaviors.
- Future plans include external tool integration and multi-turn interactions.

---

## File Structure and Main Components

```
/Users/eli/Documents/GitHub/deepsearch
│
├── README.md                 # Project overview (minimal)
├── next_action_items.md      # Notes on future agent flow improvements
├── plan_action.md            # Detailed project plan and architecture
└── agents/
    ├── agents.py             # Core multi-agent system logic (terminal interface)
    ├── gradio_agents.py      # Gradio UI wrapping the agents for web interface
    └── prompts.py            # Prompt templates for each agent
```

---

## Multi-Agent System Workflow

```text
User Query
   │
   ▼
┌───────────────────┐
│  Planning Agent   │
│  - Decomposes     │
│    query into     │
│    sub-tasks      │
└───────────────────┘
   │
   ▼
┌───────────────────┐
│  Reasoning Agent  │
│  - Expands each  │
│    sub-task with  │
│    detailed       │
│    reasoning      │
└───────────────────┘
   │
   ▼
┌─────────────────────┐
│ Confirmation Agent  │
│ - Summarizes plan   │
│   and reasoning     │
│ - Requests user     │
│   confirmation      │
└─────────────────────┘
   │
   ▼
┌───────────────────┐
│ Execution Agent   │
│ - Executes tasks  │
│ - Aggregates     │
│   results        │
└───────────────────┘
   │
   ▼
Final Report Output
```

---

## Agent Details

- **Planning Agent**  
  Uses GPT-4o to generate a JSON list of actionable sub-tasks from the user query.

- **Reasoning Agent**  
  Uses GPT-3.5-turbo to provide chain-of-thought reasoning for each sub-task.

- **Confirmation Agent**  
  Uses GPT-4o to summarize the plan and reasoning, then asks the user for approval.

- **Execution Agent**  
  Uses GPT-4o to simulate executing each task and returns results.

---

## Prompts

- Defined in `agents/prompts.py`.
- Each agent has a specific prompt template guiding its behavior.
- Prompts instruct the LLM on task decomposition, reasoning, confirmation, and execution.

---

## Interfaces

- **Terminal Interface:**  
  Implemented in `agents/agents.py`. Runs the multi-agent flow in the terminal.

- **Gradio Web UI:**  
  Implemented in `agents/gradio_agents.py`. Provides an interactive web interface for the same flow.

---

## Future Directions (from next_action_items.md and plan_action.md)

- Sub-task verification before expansion.
- Parallel processing of sub-tasks.
- Multi-turn interactions based on user feedback.
- Additional agents like Yes/No agent.
- Integration with external tools: web scrapers, summarizers, citation extractors, NER tools.
- Persistent storage migration from in-memory to SQLite or MongoDB.

---

This walkthrough should help you understand the structure and flow of the Deepsearch repo clearly.
