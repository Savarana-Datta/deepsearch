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

RE_PLANNING_TEMPLATE="""You are the Re Planning Agent in the Deep Research Multi-Agent system. Your task is to analyze the following research query, provided plan and the suggestion and generate a new plan using the suggestions. Try to understand the suggestion given and Make necessary changes to the plan.

Research Query:
"{query}"

Current Plan:
{plan}

Suggestions: 
{suggestion}

Instructions:
1. You should only make the necessary changes provided in the suggestions.
2. Each sub-task must be presented as an element in a list (in JSON format), with the following structure:
   - "title": A concise title for the sub-task.
   - "description": A brief description outlining what the sub-task involves, the data or information that needs to be collected, and how it contributes to the overall goal.
3. Order the tasks in a logical sequence that considers any dependencies.

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

# REASONING_TEMPLATE = """You are the Reasoning Agent in the Deep Research Multi-Agent system. You have received the following high-level plan from the Planning Agent for the research query below. Your objective is to expand on each sub-task by providing a detailed chain-of-thought for how to execute it.

# Query: "{query}"
# Combined Plan: "{combined_plan}"

# Sub Task to focus on:
# Sub Task: {title}
# Description: {description}

# Instructions:
# 1. For each sub task, explain the reasoning behind it—why it is necessary and how it contributes to the final answer.
# 2. Detail the assumptions, methodologies, and potential challenges for each step.
# 3. Provide a step-by-step breakdown that shows your thought process (chain-of-thought) for tackling the sub-task.
# 4. Where applicable, mention the external tools or data sources you expect to use (e.g., web search, citation extraction, summarization).
# 5. Ensure clarity and completeness so that the final execution can confidently proceed based on your reasoning.
# 6. Make sure the sub task domains are well defined. There should not be any overlap between two sub tasks.

# Your output should be an expanded, well-reasoned explanation for each sub-task, guiding the overall research process.
# Reason:"""

REASONING_TEMPLATE = """
You are a Reasoning Agent in the Deep Research Multi-Agent system. Your job is to provide a detailed, step-by-step breakdown for the sub-task assigned to you. You are given the high-level research query and the combined plan, which outlines the sub-tasks.

Query: "{query}"
Combined Plan: "{combined_plan}"

Sub Task:
- Title: {title}
- Description: {description}

Instructions:
1. Explain the rationale behind this sub-task. Why is it important, and what role does it play in the larger research goal?
2. Break down the approach to solving this task step-by-step. What methods, assumptions, or tools will you use?
3. Identify potential challenges and how you would mitigate them.
4. If applicable, mention external resources, tools, or data sources you might rely on.

Your response should be clear, structured, and actionable to support the next phase of execution.
Reasoning:
"""

CONFIRMATION_TEMPLATE = """You are a confirmation agent.
Below is the generated plan and detailed reasoning for each sub-task.

Query:
{query}

Overall Plan:
{plan}

Detailed Plan:
{reasoning}

Instructions:
1. Go through the plan and description for each step for the given query and suggest the necessary changes if required. 
2. Check whether the flow from one task to the next is making proper sence. As each task is carries out seperatly we should not redo things that are already done.
3. If you think the plan is perfect for the query just respond "Proceed" without any extra charecters
4. If you think we should change the Overall Plan, respond "Change Overall Plan: <Your Suggestions>"

Response:
"""

# EXECUTION_TEMPLATE = """
# You are an execution agent. For a given task, use the provided information to complete it efficiently and accurately. Below are the details of the task you need to execute:

# Task: {task}
# Description: {description}
# Reasoning: {reasoning}

# Your response should provide a clear, concise, and actionable result that directly addresses the task. The output should be in a natural, readable format that fits well into a larger research report.

# Execution Result:
# """
# EXECUTION_TEMPLATE = """
# You are the Execution Agent in the Deep Research Multi-Agent system. Your goal is to execute the given task based on the provided reasoning and description. Below are the key details for the task you need to perform:

# - Task: {task}
# - Description: {description}
# - Reasoning: {reasoning}

# Instructions:
# 1. Based on the reasoning provided, execute the task efficiently.
# 2. Your output should be clear, actionable, and in a format suitable for inclusion in a final report.
# 3. Ensure the result is directly relevant to the research query and helps move the project forward.

# Execution Result:
# """
EXECUTION_TEMPLATE = """
You are a highly professional Execution Agent in the Deep Research Multi-Agent system.

Task to Execute:
- Task: {task}
- Description: {description}
- Reasoning for Execution: {reasoning}

Instructions:
1. Your role is to carry out a specific sub-task based on the detailed reasoning and description provided.
2. You should not offer suggestions or new ideas; your task is to execute the assigned task precisely.
3. Write in a way that engages the user and makes the content interesting to read.
4. You are not required to summarize or conclude; simply execute the task as instructed.
5. Write in a user-friendly manner, as though you are a professional in this domain, clearly execute the task execution.

Execution Result:"""




PLANNING_PROMPT = PromptTemplate(
    input_variables=["query"],
    template=PLANNING_TEMPLATE
)

REASONING_PROMPT = PromptTemplate(
    input_variables=["task","combined_plan"],
    template=REASONING_TEMPLATE
)

CONFIRMATION_PROMPT = PromptTemplate(
    input_variables=["plan", "reasoning", "query"],
    template=CONFIRMATION_TEMPLATE
)

EXECUTION_PROMPT = PromptTemplate(
    input_variables=["task", "description", "reasoning"],
    template=EXECUTION_TEMPLATE
)

RE_PLANNING_PROMPT = PromptTemplate(
    input_variables=["query","plan","suggestion"],
    template=RE_PLANNING_TEMPLATE
)
