# Deep Research Multi-Agent

Planning to use multi-agent architecture with the following agents:

- **Planning Agent**: Decomposes the research query into a high-level plan.
- **Reasoning Agent**: Elaborates on each plan step with detailed reasoning (chain-of-thought).
- **Confirmation Agent**: Presents the plan and reasoning to the user (via terminal) for approval or modification.
- **Execution Agent**: Calls external tools to execute approved sub-tasks and compiles the final output.

> **Note:** For this initial implementation, the system will run via the terminal. Future iterations may integrate a Gradio interface, and storage will initially be in memory with a later transition to SQLite.

---

## 1. Requirements & Objectives

### Scope

- **Input:** Text-based research queries.
- **Output:** A comprehensive answer built through multi-step planning, reasoning, and execution(along with citations if possible).
- **Metrics:** Should think..!

### Agent Roles & Responsibilities

- **Planning Agent:**  
  - Receives the user query.
  - Generates a high-level plan (a list of sub-tasks) to tackle the query.

- **Reasoning Agent:**  
  - Expands each sub-task with detailed reasoning steps.
  - Uses chain-of-thought prompting to provide clarity.

- **Confirmation Agent:**  
  - Presents the combined plan and reasoning to the user (via terminal and then via gradio).
  - Accepts user confirmation, modifications (Feedback can be given to reasoning or planning agent depending on severity), or rejections.
  
- **Execution Agent:**  
  - Maps each confirmed sub-task to appropriate external tools.
  - Calls the tools, handles retries (retry up to n times before returning empty), and aggregates outputs.

---

## 2. System Architecture

### Overview

- **Multi-Agent Design:**  
  - Each agent (Planning, Reasoning, Confirmation, Execution) is a separate module.
  - Agents communicate via a shared in-memory storage mechanism (later upgrade it to SQLite/Mongo).

- **Control Flow:**  
  1. **User Input:** get the query from user.
  2. **Planning Phase:** The Planning Agent generates a high-level plan.
  3. **Reasoning Phase:** The Reasoning Agent adds detailed reasoning to each plan step.
  4. **Confirmation Phase:** The Confirmation Agent displays the plan and reasoning for user review via terminal.
  5. **Execution Phase:** The Execution Agent calls external tools to execute confirmed tasks.
  6. **Iteration:** If the user requests modifications, the system loops back to the relevant phase. 
        - should be done ideally at step 4. 

### Inter-Agent Communication

- **Shared Memory:**  
  - For now, use an in-memory storage structure (a dictionary or json) to store plans, reasoning outputs, and user confirmations.
  - This storage will later be migrated to SQLite for persistence.

---

## 3. Tools & External API Integration

### List of Potential Tools

We need several text-based tools:

- **Web Search Tool:**  
  - Text-based search API.

- **Document Summarization Tool:**  
  - An API endpoint for summarizing long articles.
  - Or we can have another agent for the same.

- **Citation Extraction & Verification Tool:**  
  - Only for the research work. We can skip in phase 1 and get back to this in phase 2.
  - API or parser to extract and verify citation details from fetched articles.

- **Named Entity Recognition (NER) Tool:**  
  - Utilize libraries (such as spaCy or Hugging Face models) for identifying key entities. 
  - Help's us in the step 3 (Reasoning Phase) for more complex tasks. Get the labels using this tool and then get the reasoning.
  - Use small model for this and plan when to use and when not to. We might save some time here.

- **LLM endpoints:**  
  - Have all the llm endpoint calling functions in one module.

- **Retry Mechanism:**  
  - Implement a function that retries API calls up to a configurable n times, then returns empty if all attempts fail.