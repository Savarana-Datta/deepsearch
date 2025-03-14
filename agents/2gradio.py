# agents.py
import json
import os
import requests  # For fetching URLs in parallel
from bs4 import BeautifulSoup  # For parsing HTML content
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import PLANNING_PROMPT, REASONING_PROMPT, CONFIRMATION_PROMPT, EXECUTION_PROMPT, RE_PLANNING_PROMPT
from concurrent.futures import ThreadPoolExecutor
from tools import *
from pprint import pprint
import gradio as gr  # Import Gradio

load_dotenv()

# Get OpenAI API key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize shared LLM instances
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
            return []
        return tasks

    def generate_plan(self, query):
        result = self.chain.run(query=query)
        plan = self.parse_plan_json(result)
        return plan

class ReasoningAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_o3, prompt=REASONING_PROMPT)
    
    def generate_reasoning(self, task, query, combined_plan):
        reasoning = self.chain.run(title=task['title'], description=task['description'], query=query, combined_plan=combined_plan)
        return reasoning.strip()

class ConfirmationAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=CONFIRMATION_PROMPT)
    
    def get_confirmation(self, plan, query):
        plan_text = [pl['title'] for pl in plan]
        plan_text = '\n'.join(plan_text)
        reasoning_text = ["Task:" + pl['title'] + '\n' + "Detailed Steps:" + pl.get('reasoning', '') for pl in plan]
        reasoning_text = '\n'.join(reasoning_text)
        print("Detailed Plan:\n", reasoning_text)
        response_from_confirmation_agent = self.chain.run(plan=plan_text, reasoning=reasoning_text, query=query)
        print("Confirmation Agent Response:\n", response_from_confirmation_agent)
        return response_from_confirmation_agent.strip()

class ExecutionAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=EXECUTION_PROMPT)
    
    def execute_task(self, task_title, task_description, task_reasoning, retries=3):
        search_query = task_description  # Use task description as the search query
        print("Performing web search for:", search_query)
        try:
            web_data = get_web_content_from_query(search_query, num_results=6)
            # Helper function to fetch and/or process text concurrently.
            def fetch_text(item):
                if 'text' in item and item['text']:
                    return item['text'][:500]
                elif 'url' in item:
                    try:
                        response = requests.get(item['url'], timeout=5)
                        if response.ok:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            return soup.get_text()[:500]
                        else:
                            return ""
                    except Exception as e:
                        print("Error fetching URL:", item['url'], e)
                        return ""
                else:
                    return ""
            with ThreadPoolExecutor() as executor:
                texts = list(executor.map(fetch_text, web_data))
            combined_content = "\n\n".join(text for text in texts if text)
            task_reasoning += "\n\nWeb Search Context:\n" + combined_content
        except Exception as e:
            print("Error during web scraping:", e)
        
        for attempt in range(retries):
            result = self.chain.run(task=task_title, description=task_description, reasoning=task_reasoning)
            if result and result.strip():
                return result.strip()
        return f"Task: {task_title} -> No result after {retries} retries."

class RePlanningAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=RE_PLANNING_PROMPT)
    def parse_plan_json(self, json_str):
        try:
            start_index = json_str.find('[')
            end_index = json_str.rfind(']') + 1
            if start_index != -1 and end_index != -1:
                json_str = json_str[start_index:end_index]
            tasks = json.loads(json_str)
        except json.JSONDecodeError as e:
            return []
        return tasks

    def generate_plan(self, query, plan, suggestion):
        result = self.chain.run(query=query, plan=plan, suggestion=suggestion)
        plan = self.parse_plan_json(result)
        return plan

def run_pipeline(query):
    # Instantiate agents
    planning_agent = PlanningAgent()
    reasoning_agent = ReasoningAgent()
    confirmation_agent = ConfirmationAgent()
    execution_agent = ExecutionAgent()
    replanning_agent = RePlanningAgent()
    
    # --- Phase 1: Planning ---
    plan = planning_agent.generate_plan(query)
    high_level_output = "" + json.dumps(plan, indent=2) + "\n\n"
    yield high_level_output, "", ""
    
    # --- Phase 2: Detailed Reasoning ---
    combined_plan = ["Task:" + pl['title'] + '\n' + "Description:" + pl['description'] for pl in plan]
    combined_plan = '\n'.join(combined_plan)
    
    def generate_reasoning_for_task(task):
        return reasoning_agent.generate_reasoning(task, query, combined_plan)
    
    with ThreadPoolExecutor() as executor:
        reasoning_results = list(executor.map(generate_reasoning_for_task, plan))
    
    reasoning_output = ""
    for idx, reasoning in enumerate(reasoning_results):
        plan[idx]['reasoning'] = reasoning
        reasoning_output += f"Task: {plan[idx]['title']}\nReasoning: {reasoning}\n\n"
    yield high_level_output, reasoning_output, ""
    
    # --- Phase 3: Confirmation & Execution ---
    confirmation = confirmation_agent.get_confirmation(plan, query)
    attempts = 0
    while confirmation.strip() != 'Proceed' and attempts < 2:
        suggestion = confirmation.split(':')[1] if ':' in confirmation else ""
        plan = replanning_agent.generate_plan(query, plan, suggestion)
        confirmation = confirmation_agent.get_confirmation(plan, query)
        attempts += 1
    
    # Execute each task and stream the final report in order.
    final_report_output = ""
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(execution_agent.execute_task, task['title'], task['description'], task['reasoning']) for task in plan]
        final_results = [None] * len(futures)
        for idx, future in enumerate(futures):
            final_results[idx] = future.result()  # Block until the result is ready (in order)
            final_report_output = "" + "\n".join(final_results[:idx+1])
            yield high_level_output, reasoning_output, final_report_output

# Create a Gradio interface with three output boxes.
iface = gr.Interface(
    fn=run_pipeline,
    inputs=gr.Textbox(lines=2, placeholder="Enter your research query here..."),
    outputs=[
        gr.Textbox(label="High-level Plan"),
        gr.Textbox(label="Detailed Reasoning Plan"),
        gr.Textbox(label="Final Answer")
    ],
    title="Multi-Agent Research Pipeline",
    description="Enter your research query and hit submit. The final report will stream as each task completes in order."
)

if __name__ == "__main__":
    iface.launch(share=True)
