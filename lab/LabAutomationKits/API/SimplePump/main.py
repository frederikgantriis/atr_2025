from fastapi import FastAPI
from driver import RealMicrocontrollerService, get_logger

import time
from typing import Set
import json

from action_models import ActionRequest

import asyncio
import os
import httpx

app = FastAPI()
micro = RealMicrocontrollerService()

log = get_logger(__name__)
busy = False
stop_requested = False

# --- Helper methods ---
def send_command_to_hardware(pumpA, duration):
    # Extract pump parameters
    stateA = pumpA.state if pumpA else False
    speedA = pumpA.speed if pumpA else 0
    dirA = pumpA.dir if pumpA else True
    
    
    # Send command to hardware
    micro.set_state(
        stateA, speedA, dirA,
        duration
    )
    
    return duration

# --- Unified background task ---
def action_task(request: ActionRequest):
    job_id = request.id
    pumpA = request.pumpA
    duration = request.time
      # Send motor command
    step_time = send_command_to_hardware(pumpA, duration)

@app.post("/actions")
async def perform_actions(request: ActionRequest):
    global busy, stop_requested
    if busy:
        return {"status": "busy"}
    busy = True
    stop_requested = False
    # Send 'acknowledged' webhook immediately
    
    action_task(request)
    return {"status": "accepted", "id": request.id}

@app.post("/stop")
async def emergency_stop():
    global stop_requested, busy
    stop_requested = True
    busy = False  # Fallback: ensure busy is reset if stop is called
    log.info("/stop called: busy set to False")
    return {"status": "stopped"}


@app.get("/status")
def get_status():
    global busy
    return {"busy": busy} 

