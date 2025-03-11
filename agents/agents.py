import json
import os
from dotenv import load_dotenv
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from prompts import PLANNING_PROMPT, REASONING_PROMPT, CONFIRMATION_PROMPT, EXECUTION_PROMPT
from pprint import pprint

# Load environment variables from .env file
load_dotenv()

# Get Fireworks API key and URL from .env file
# Load environment variables from .env file
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
            pprint(tasks)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            return []
        return tasks

    def generate_plan(self, query):
        print('Now')
        result = self.chain.run(query=query)
        print("Here",result)
        plan = self.parse_plan_json(result)
        return plan

class ReasoningAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_o3, prompt=REASONING_PROMPT)
    
    def generate_reasoning(self, task, query):
        reasoning = self.chain.run(title=task['title'], description=task['description'], query=query)
        return reasoning.strip()

class ConfirmationAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=CONFIRMATION_PROMPT)
    
    def get_confirmation(self, plan):
        plan_text = [pl['title'] for pl in plan]
        plan_text = '\n'.join(plan_text)
        reasoning_text = [pl['reasoning'] for pl in plan]
        reasoning_text = '\n'.join(reasoning_text)
        summary = self.chain.run(plan=plan_text, reasoning=reasoning_text)
        print("Confirmation Summary:\n", summary)
        user_input = input("Is the plan acceptable? (Yes/No): ")
        return user_input.strip().lower() == "yes"

class ExecutionAgent:
    def __init__(self):
        self.chain = LLMChain(llm=llm_4o, prompt=EXECUTION_PROMPT)
    
    def execute_task(self, task, retries=3):
        # Implement a simple retry mechanism
        for attempt in range(retries):
            result = self.chain.run(task=task)
            if result and result.strip():
                return f"Task: {task} -> {result.strip()}"
        return f"Task: {task} -> No result after {retries} retries."

def main():
    query = input("Enter your research query: ")

    # Instantiate agents
    planning_agent = PlanningAgent()
    reasoning_agent = ReasoningAgent()
    confirmation_agent = ConfirmationAgent()
    execution_agent = ExecutionAgent()

    # Phase 1: Planning
    plan = planning_agent.generate_plan(query)
    # Phase 2: Reasoning
    # reasoning_list = []
    for task in plan:
        reasoning = reasoning_agent.generate_reasoning(task, query)
        task['reasoning'] = reasoning
    print("\nGenerated Reasoning:")
    for idx, reasoning in enumerate(plan, start=1):
        print(f"Task {reasoning['title']} Reasoning: {reasoning['reasoning']}")

    # Phase 3: Confirmation
    if not confirmation_agent.get_confirmation(plan):
        print("Plan not confirmed. Exiting.")
        return

    # Phase 4: Execution
    results = []
    for task in plan:
        result = execution_agent.execute_task(task)
        results.append(result)

    # Final output
    print("\nFinal Report:")
    for res in results:
        print(res)

if __name__ == "__main__":
    main()