import json
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import PLANNING_PROMPT, REASONING_PROMPT, CONFIRMATION_PROMPT, EXECUTION_PROMPT
from pprint import pprint
import gradio as gr

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm_4o = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.5,
    openai_api_key=openai_api_key,
    max_tokens=3200
)
llm_o3 = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    openai_api_key=openai_api_key,
    max_tokens=3200
)

class PlanningAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=PLANNING_PROMPT)
    
    def parse_plan_json(self, json_str):
        try:
            start_index = json_str.find('[')
            end_index = json_str.rfind(']') + 1
            if start_index != -1 and end_index != -1:
                json_str = json_str[start_index:end_index]
            tasks = json.loads(json_str)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            return []
        return tasks
    
    def generate_plan(self, query):
        result = self.chain.run(query=query)
        plan = self.parse_plan_json(result)
        return plan

class ReasoningAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_o3, prompt=REASONING_PROMPT)
    
    def generate_reasoning(self, task, query):
        return self.chain.run(title=task['title'], description=task['description'], query=query).strip()

class ConfirmationAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=CONFIRMATION_PROMPT)
    
    def get_confirmation_summary(self, plan):
        plan_text = "\n".join([pl['title'] for pl in plan])
        reasoning_text = "\n".join([pl['reasoning'] for pl in plan])
        return self.chain.run(plan=plan_text, reasoning=reasoning_text)

class ExecutionAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=EXECUTION_PROMPT)
    
    def execute_task(self, task):
        return self.chain.run(task=task)

def generate_plan_interface(query):
    planning_agent = PlanningAgent()
    reasoning_agent = ReasoningAgent()
    confirmation_agent = ConfirmationAgent()
    
    # Phase 1: Generate plan and reasoning for each task
    plan = planning_agent.generate_plan(query)
    for task in plan:
        task['reasoning'] = reasoning_agent.generate_reasoning(task, query)
    
    # Get confirmation summary text from the confirmation agent
    confirmation_summary = confirmation_agent.get_confirmation_summary(plan)
    
    # Format the plan for display
    plan_text = "\n".join([f"{t['title']}: {t['reasoning']}" for t in plan])
    # Return the plan display, confirmation summary, and the plan as JSON (for hidden state)
    return plan_text, confirmation_summary, json.dumps(plan)

def confirm_and_execute(confirmation_choice, plan_json):
    if confirmation_choice.lower() != "yes":
        return "Plan not confirmed. Exiting."
    
    plan = json.loads(plan_json)
    execution_agent = ExecutionAgent()
    results = []
    for task in plan:
        result = execution_agent.execute_task(task)
        results.append(f"Task '{task['title']}': {result.strip()}")
    final_report = "\n".join(results)
    return final_report

with gr.Blocks() as demo:
    gr.Markdown("### Research Query Input")
    query_input = gr.Textbox(label="Enter your research query")
    
    with gr.Row():
        generate_button = gr.Button("Generate Plan")
    
    plan_output = gr.Textbox(label="Generated Plan and Reasoning", interactive=False)
    confirmation_summary_output = gr.Textbox(label="Confirmation Summary", interactive=False)
    
    # Hidden state to store the plan as JSON
    plan_state = gr.State()
    
    generate_button.click(fn=generate_plan_interface, inputs=query_input, 
                          outputs=[plan_output, confirmation_summary_output, plan_state])
    
    gr.Markdown("### Confirm Plan")
    confirmation_input = gr.Radio(choices=["Yes", "No"], label="Is the plan acceptable?")
    execute_button = gr.Button("Execute Tasks")
    execution_output = gr.Textbox(label="Final Output", interactive=False)
    
    execute_button.click(fn=confirm_and_execute, inputs=[confirmation_input, plan_state],
                         outputs=execution_output)

demo.launch(share=True)
