import gradio as gr
import json
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import PLANNING_PROMPT, REASONING_PROMPT, CONFIRMATION_PROMPT, EXECUTION_PROMPT, RE_PLANNING_PROMPT
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI LLM instances
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
            # Parse the JSON string into a Python list
            tasks = json.loads(json_str)
            
            # Iterate over each task and print its details
            # pprint(tasks)
        except json.JSONDecodeError as e:
            # print("Error parsing JSON:", e)
            return []
        return tasks

    def generate_plan(self, query):
        # print('Now')
        result = self.chain.run(query=query)
        # print("Here",result)
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
        reasoning_text = ["Task:"+ pl['title']+ '\n'+"Detailed Steps:"+pl['reasoning'] for pl in plan]
        reasoning_text = '\n'.join(reasoning_text)
        print("Detailed Plan:\n",reasoning_text)
        response_from_confirmation_agent = self.chain.run(plan=plan_text, reasoning=reasoning_text, query=query)
        print("Confirmation Agent Response:\n", response_from_confirmation_agent)
        # user_input = input("Is the plan acceptable? (Yes/No): ")
        return response_from_confirmation_agent.strip()

class ExecutionAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=EXECUTION_PROMPT)
    
    def execute_task(self, task_title, task_description, task_reasoning, retries=3):
        # Implement a simple retry mechanism
        for attempt in range(retries):
            result = self.chain.run(task=task_title, description=task_description, reasoning=task_reasoning)
            if result and result.strip():
                # return f"Task: {task} -> {result.strip()}"
                return result.strip()
        # return f"Task: {task} -> No result after {retries} retries."
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
            # Parse the JSON string into a Python list
            tasks = json.loads(json_str)
            
            # Iterate over each task and print its details
            # pprint(tasks)
        except json.JSONDecodeError as e:
            # print("Error parsing JSON:", e)
            return []
        return tasks

    def generate_plan(self, query, plan, suggestion):
        # print('Now')
        result = self.chain.run(query=query, plan=plan, suggestion=suggestion)
        # print("Here",result)
        plan = self.parse_plan_json(result)
        return plan


def generate_results(query):
    planning_agent = PlanningAgent()
    reasoning_agent = ReasoningAgent()
    confirmation_agent = ConfirmationAgent()
    execution_agent = ExecutionAgent()
    replanning_agent = RePlanningAgent()
    
    def generate_reasoning_for_task( task, query, combined_plan):
        return reasoning_agent.generate_reasoning(task, query, combined_plan)

    def execute_task_for_plan(task):
        task_title = task['title']
        task_description = task['description']
        task_reasoning = task['reasoning']  
        return execution_agent.execute_task(task_title, task_description, task_reasoning)

    # Phase 1: Planning
    plan = planning_agent.generate_plan(query)
    print("Initial Plan:", plan)
    # Phase 2: Reasoning
    combined_plan = ["Task:"+ pl['title']+ '\n'+"Description:"+pl['description'] for pl in plan]
    combined_plan = '\n'.join(combined_plan)
    
    with ThreadPoolExecutor() as executor:
        reasoning_results = list(executor.map(generate_reasoning_for_task, plan, [query] * len(plan), [combined_plan] * len(plan)))
    
    resoning_string = ''
    for idx, reasoning in enumerate(reasoning_results):
        plan[idx]['reasoning'] = reasoning
        resoning_string = resoning_string+ plan[idx]['title']+'\n'+plan[idx]['description']+'\n'+plan[idx]['reasoning'] + '\n'
    
    print("\nGenerated Reasoning:")
    for idx, reasoning in enumerate(plan, start=1):
        print(f"Task: {reasoning['title']} \nReasoning: {reasoning['reasoning']}")
        
    # Phase 3: Confirmation
    confirmation = confirmation_agent.get_confirmation(plan,query)
    while(confirmation.strip()!='Proceed'):
        print("Plan not confirmed. Rerunning.")
        suggestion = confirmation.split(':')[1]
        plan = replanning_agent.generate_plan(query,plan,suggestion)
        confirmation = confirmation_agent.get_confirmation(plan,query)
        
    # Phase 4: Execution
    with ThreadPoolExecutor() as executor:
        execution_results = list(executor.map(execute_task_for_plan, plan))
        
    # Final output
    print("\nFinal Report:")
    final_string = ''
    for res in execution_results:
        final_string = final_string+'\n'+res
        print(res)

    return plan, resoning_string, final_string

# Define Gradio interface
def gradio_interface(query):
    # Ensure the query is being processed correctly
    initial_plan, reasoning_output, final_report = generate_results(query)
    
    # Convert the plan to a string for display
    plan_output = "\n\n".join([f"Task: {task['title']}\nDescription: {task['description']}" for task in initial_plan])
    
    # Return all information immediately as per the query processing
    return plan_output, reasoning_output, final_report

# Gradio UI components
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Enter your research query:", lines=2),
    outputs=[
        gr.Textbox(label="Initial Plan", lines=10),
        gr.Textbox(label="Task Reasoning", lines=10),
        gr.Textbox(label="Final Execution Report", lines=15)
    ],
    live=False  # We want the results only after the query is processed, not live updates
)

if __name__ == "__main__":
    iface.launch(share=True)
