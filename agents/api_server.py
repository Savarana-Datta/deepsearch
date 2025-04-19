from fastapi import FastAPI, BackgroundTasks, HTTPException, Body
from pydantic import BaseModel
import uvicorn
import json
from typing import List, Dict, Any, Optional
from uuid import uuid4
import asyncio
from agents import PlanningAgent, ReasoningAgent, ExecutionAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deep Research Multi-Agent API",
    description="API for Deep Research Multi-Agent system with sequential thinking",
    version="1.0.0"
)

# In-memory storage for jobs
jobs = {}

class QueryRequest(BaseModel):
    query: str
    max_steps: int = 5  # Default maximum number of thinking steps

class SequentialThinkingStep(BaseModel):
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: bool = False

class JobStatus(BaseModel):
    job_id: str
    status: str
    current_step: str
    progress: int
    result: Optional[Dict[str, Any]] = None
    thinking_steps: List[SequentialThinkingStep] = []
    error: Optional[str] = None

@app.post("/research", response_model=Dict[str, str])
async def create_research_task(request: QueryRequest):
    job_id = str(uuid4())
    
    # Initialize job data
    jobs[job_id] = {
        "status": "queued",
        "current_step": "initialization",
        "progress": 0,
        "query": request.query,
        "max_steps": request.max_steps,
        "thinking_steps": [],
        "result": None,
        "error": None
    }
    
    # Start processing in background
    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_research_task, job_id)
    
    return {"job_id": job_id, "message": "Research task has been queued"}

async def process_research_task(job_id: str):
    try:
        # Get job details
        job = jobs[job_id]
        query = job["query"]
        max_steps = job["max_steps"]
        
        jobs[job_id]["status"] = "running"
        jobs[job_id]["current_step"] = "planning"
        
        # Initialize agents
        planning_agent = PlanningAgent()
        reasoning_agent = ReasoningAgent()
        execution_agent = ExecutionAgent()
        
        # Phase 1: Sequential Thinking for Planning
        thinking_step = SequentialThinkingStep(
            thought="Analyzing the research query to identify the main components and necessary tasks.",
            thought_number=1,
            total_thoughts=max_steps,
            next_thought_needed=True
        )
        jobs[job_id]["thinking_steps"].append(thinking_step.dict())
        jobs[job_id]["progress"] = 10
        
        # Generate initial plan
        plan = planning_agent.generate_plan(query)
        
        # Phase 2: Sequential Thinking for each plan step
        thinking_number = 2
        for i, task in enumerate(plan):
            # Update job status
            jobs[job_id]["current_step"] = f"reasoning_for_task_{i+1}"
            jobs[job_id]["progress"] = 10 + (i+1) * (40 // len(plan))
            
            # Add reasoning thinking step
            thinking_step = SequentialThinkingStep(
                thought=f"Developing reasoning for task: {task['title']}. Understanding how this contributes to the overall research goal.",
                thought_number=thinking_number,
                total_thoughts=max_steps,
                next_thought_needed=(thinking_number < max_steps)
            )
            jobs[job_id]["thinking_steps"].append(thinking_step.dict())
            
            # Generate reasoning
            reasoning = reasoning_agent.generate_reasoning(task, query)
            task['reasoning'] = reasoning
            thinking_number += 1
            
            # If we've reached max steps, break
            if thinking_number > max_steps:
                break
        
        # Phase 3: Sequential Thinking for Execution
        jobs[job_id]["current_step"] = "execution"
        jobs[job_id]["progress"] = 60
        
        results = []
        for i, task in enumerate(plan):
            if thinking_number <= max_steps:
                thinking_step = SequentialThinkingStep(
                    thought=f"Executing task: {task['title']}. Synthesizing information and generating results.",
                    thought_number=thinking_number,
                    total_thoughts=max_steps,
                    next_thought_needed=(thinking_number < max_steps)
                )
                jobs[job_id]["thinking_steps"].append(thinking_step.dict())
                
            # Execute task
            result = execution_agent.execute_task(task)
            results.append(result)
            
            # Update job progress
            jobs[job_id]["progress"] = 60 + (i+1) * (30 // len(plan))
            thinking_number += 1
            
            # If we've reached max steps, break
            if thinking_number > max_steps:
                break
        
        # Final step - synthesis
        if thinking_number <= max_steps:
            thinking_step = SequentialThinkingStep(
                thought="Synthesizing all findings and formulating a comprehensive answer to the original research query.",
                thought_number=thinking_number,
                total_thoughts=max_steps,
                next_thought_needed=False
            )
            jobs[job_id]["thinking_steps"].append(thinking_step.dict())
        
        # Complete the job
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["current_step"] = "completed"
        jobs[job_id]["result"] = {
            "plan": plan,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["progress"] = 100

@app.get("/research/{job_id}", response_model=JobStatus)
async def get_research_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job_data["status"],
        current_step=job_data["current_step"],
        progress=job_data["progress"],
        result=job_data["result"],
        thinking_steps=job_data["thinking_steps"],
        error=job_data["error"]
    )

@app.post("/research/{job_id}/think", response_model=SequentialThinkingStep)
async def add_thinking_step(
    job_id: str, 
    thinking: SequentialThinkingStep = Body(...)
):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]["status"] == "completed":
        raise HTTPException(status_code=400, detail="Job is already completed")
    
    # Add the new thinking step
    jobs[job_id]["thinking_steps"].append(thinking.dict())
    
    # If this is the final thinking step, update status
    if not thinking.next_thought_needed:
        jobs[job_id]["status"] = "awaiting_execution"
    
    return thinking

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)