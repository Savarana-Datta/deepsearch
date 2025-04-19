# Execution Agent Walkthrough

This document provides a focused overview of the Execution Agent in the Deepsearch multi-agent system, illustrated with a box and arrow schema for clarity.

---

## Execution Agent Role

The Execution Agent is responsible for:

- Receiving confirmed sub-tasks from the Confirmation Agent.
- Simulating the execution of each task using a language model (GPT-4o).
- Returning the results of each task execution.
- Aggregating all task results into a final report.

---

## Execution Agent Workflow

```text
Confirmed Sub-Tasks
       │
       ▼
┌───────────────────────┐
│   Execution Agent     │
│ ┌───────────────────┐ │
│ │ 1. Receive Task   │ │
│ ├───────────────────┤ │
│ │ 2. Run LLMChain   │ │
│ │    with EXECUTION │ │
│ │    PROMPT         │ │
│ ├───────────────────┤ │
│ │ 3. Retry if no    │ │
│ │    valid output   │ │
│ │    (up to 3 times)│ │
│ ├───────────────────┤ │
│ │ 4. Return Result  │ │
│ └───────────────────┘ │
└───────────────────────┘
       │
       ▼
Final Report Output
```

---

## Implementation Details

- Uses GPT-4o model via LangChain's LLMChain.
- The EXECUTION_PROMPT instructs the model to simulate task execution.
- Implements a retry mechanism to ensure valid responses.
- Aggregates results for all tasks into a comprehensive final report.

---

This schema and explanation clarify the Execution Agent's function within the Deepsearch system.
