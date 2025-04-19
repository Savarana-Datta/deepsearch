from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Generator
from agents import agents
import uvicorn
import json

app = FastAPI(title="Deepsearch Multi-Agent API")

class Task(BaseModel):
    title: str
    description: str
    reasoning: str = None

class QueryRequest(BaseModel):
    query: str

class ConfirmRequest(BaseModel):
    confirmed: bool
    plan: List[Task]

# Instantiate agents with streaming enabled
planning_agent = agents.PlanningAgent()
reasoning_agent = agents.ReasoningAgent()
execution_agent = agents.ExecutionAgent()

def stream_plan_and_reasoning(query: str) -> Generator[str, None, None]:
    # Stream planning output
    plan = planning_agent.generate_plan(query)
    yield json.dumps({"type": "plan", "data": plan}) + "\n"
    # Stream reasoning for each task
    for task in plan:
        reasoning = reasoning_agent.generate_reasoning(task, query)
        task['reasoning'] = reasoning
        yield json.dumps({"type": "reasoning", "task": task['title'], "data": reasoning}) + "\n"

def stream_execution(plan: List[Dict[str, Any]]) -> Generator[str, None, None]:
    for task in plan:
        result = execution_agent.execute_task(task)
        yield json.dumps({"type": "execution", "task": task['title'], "data": result}) + "\n"

@app.post("/query")
def generate_plan(request: QueryRequest):
    try:
        return StreamingResponse(stream_plan_and_reasoning(request.query), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/confirm")
def confirm_and_execute(request: ConfirmRequest):
    if not request.confirmed:
        return {"results": ["Plan not confirmed. Execution aborted."]}
    try:
        plan = [task.dict() for task in request.plan]
        return StreamingResponse(stream_execution(plan), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
