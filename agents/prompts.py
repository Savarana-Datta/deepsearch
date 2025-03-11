from langchain.prompts import PromptTemplate

PLANNING_TEMPLATE = """You are the Planning Agent in the Deep Research Multi-Agent system. Your task is to analyze the following research query and generate a comprehensive plan, broken down into a list of distinct, actionable sub-tasks.

Research Query:
"{query}"

Instructions:
1. Decompose the research query into a series of clear and distinct sub-tasks.
2. Each sub-task must be presented as an element in a list (in JSON format), with the following structure:
   - "title": A concise title for the sub-task.
   - "description": A brief description outlining what the sub-task involves, the data or information that needs to be collected, and how it contributes to the overall goal.
3. Order the tasks in a logical sequence that considers any dependencies.
4. Ensure the plan covers all necessary aspects to gather the required data for a comprehensive answer.

Output Format:
[
  {{
    "title": "Task Title 1",
    "description": "Detailed description of task 1."
  }},
  {{
    "title": "Task Title 2",
    "description": "Detailed description of task 2."
  }}
]
"""

REASONING_TEMPLATE = """You are the Reasoning Agent in the Deep Research Multi-Agent system. You have received the following high-level plan from the Planning Agent for the research query below. Your objective is to expand on each sub-task by providing a detailed chain-of-thought for how to execute it.

Research Query:
"{query}"

Sub Task from the Planning Agent:
Title: {title}
Description: {description}

Instructions:
1. For each sub-task, explain the reasoning behind it—why it is necessary and how it contributes to the final answer.
2. Detail the assumptions, methodologies, and potential challenges for each step.
3. Provide a step-by-step breakdown that shows your thought process (chain-of-thought) for tackling the sub-task.
4. Where applicable, mention the external tools or data sources you expect to use (e.g., web search, citation extraction, summarization).
5. Ensure clarity and completeness so that the final execution can confidently proceed based on your reasoning.

Your output should be an expanded, well-reasoned explanation for each sub-task, guiding the overall research process.
Reason:"""

CONFIRMATION_TEMPLATE = """You are a confirmation agent.
Below is the generated plan and detailed reasoning for each sub-task.

Plan:
{plan}

Reasoning:
{reasoning}

Provide a summary for the user to review and ask: "Is this plan acceptable? Answer with 'Yes' or 'No'." 
Response:"""

EXECUTION_TEMPLATE = """You are an execution agent.
Given the task: {task}
Simulate executing the task and return the result.
Execution:"""

PLANNING_PROMPT = PromptTemplate(
    input_variables=["query"],
    template=PLANNING_TEMPLATE
)

REASONING_PROMPT = PromptTemplate(
    input_variables=["task"],
    template=REASONING_TEMPLATE
)

CONFIRMATION_PROMPT = PromptTemplate(
    input_variables=["plan", "reasoning"],
    template=CONFIRMATION_TEMPLATE
)

EXECUTION_PROMPT = PromptTemplate(
    input_variables=["task"],
    template=EXECUTION_TEMPLATE
)
