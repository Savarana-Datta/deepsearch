# agents.py
import json
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import PLANNING_PROMPT, REASONING_PROMPT, CONFIRMATION_PROMPT, EXECUTION_PROMPT, RE_PLANNING_PROMPT
from concurrent.futures import ThreadPoolExecutor

from tools import *

from pprint import pprint
# from langchain_community.llms import Together
from together import Together

load_dotenv()

# Get OpenAI API key from .env file
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize a shared LLM instance using OpenAI GPT-4o or GPT-3.5-turbo
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

together_api_key = os.getenv("TOGETHER_AI")

class PlanningAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_o3, prompt=PLANNING_PROMPT)
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
        # Check if task requires web scraping based on keywords
        # if any(keyword in task_description.lower() for keyword in ["web search", "scrape", "links"]):
            # Use the task description or an appended query as search criteria
        search_query = task_description  # or extract a specific search query from your plan
        print("Performing web search for:", search_query)
        try:
            web_data = get_web_content_from_query(search_query, num_results=6)
            # Combine content from all links (limit size if necessary)
            combined_content = "\n\n".join(item['text'][:500] for item in web_data if item['text'])
            task_reasoning += "\n\nWeb Search Context:\n" + combined_content
        except Exception as e:
            print("Error during web scraping:", e)
        
        # Execute the task with the enhanced reasoning context
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

def main():
    query = input("Enter your research query: ")

    # Instantiate agents
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
    # reasoning_list = []
    combined_plan = ["Task:"+ pl['title']+ '\n'+"Description:"+pl['description'] for pl in plan]
    combined_plan = '\n'.join(combined_plan)
    
    # for task in plan:
    #     reasoning = reasoning_agent.generate_reasoning(task, query, combined_plan)
    #     task['reasoning'] = reasoning
    with ThreadPoolExecutor() as executor:
        reasoning_results = list(executor.map(generate_reasoning_for_task, plan, [query] * len(plan), [combined_plan] * len(plan)))
        
    for idx, reasoning in enumerate(reasoning_results):
        plan[idx]['reasoning'] = reasoning
    
    print("\nGenerated Reasoning:")
    for idx, reasoning in enumerate(plan, start=1):
        print(f"Task: {reasoning['title']} \nReasoning: {reasoning['reasoning']}")
        
    confirmation = confirmation_agent.get_confirmation(plan,query)
    while(confirmation.strip()!='Proceed'):
        print("Plan not confirmed. Rerunning.")
        suggestion = confirmation.split(':')[1]
        plan = replanning_agent.generate_plan(query,plan,suggestion)
        confirmation = confirmation_agent.get_confirmation(plan,query)

    # Phase 3: Confirmation
    # if not confirmation_agent.get_confirmation(plan, query):
    #     print("Plan not confirmed. Exiting.")
    #     return

    # Phase 4: Execution
    with ThreadPoolExecutor() as executor:
        # print(plan)
        # execution_results = list(executor.map(execute_task_for_plan, [query]*len(plan), [combined_plan]*len(plan), plan))
        execution_results = list(executor.map(execute_task_for_plan, plan))
        
    # Final output
    print("\nFinal Report:")
    for res in execution_results:
        print(res)
        
    # def save_results_to_file(execution_results, filename="execution_results.txt"):
    #     with open(filename, "w") as file:
    #         for result in execution_results:
    #             file.write(result + "\n")
                
    # save_results_to_file(execution_results)

if __name__ == "__main__":
    main()
