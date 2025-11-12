from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException
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

main_event_loop = None

@app.on_event("startup")
async def startup_event():
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()

@app.post("/actions")
async def perform_actions(request: ActionRequest, background_tasks: BackgroundTasks):
    global busy, stop_requested
    if busy:
        return {"status": "busy"}
    busy = True
    stop_requested = False
    # Send 'acknowledged' webhook immediately
    
    background_tasks.add_task(action_task, request)
    return {"status": "accepted", "id": request.id}

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

def handle_stop(job_id):
    """Handles stopping everything, fetching/logging/streaming state, and sending notifications."""
    micro.stop_all()  # Unified stop for all motors (pumps and mixer)
    try:
        current_state = micro.getState()
        log.info(f"Fetched state after stopping everything: {current_state}")
        state_msg = json.dumps({
            "event": "state_update",
            "state": current_state,
            "timestamp": time.time()
        })
    except Exception as e:
        log.error(f"Error fetching state after stopping everything: {e}")

def monitor_operations(job_id, pumpA, step_time):
    global busy, stop_requested
    pump_done = False if (pumpA) else True
    start_time = time.time()
    
    interval = 0.1


    while not (pump_done):
        now = time.time()
        elapsed = now - start_time
        if stop_requested:
            busy = False
            log.info("monitor_operations: busy set to False after stop_requested")
            # Send 'stopped' webhook
            return  # Exit the function gracefully after handling stop
        # --- HARD TIMEOUT ---
        if elapsed >= ((step_time / 1000)+1):
            log.info("monitor_operations: hard timeout reached, forcing completion")
            if not pump_done:
                pump_done = True
            break
        # --- NORMAL COMPLETION ---
        if not pump_done and micro.check_for_step_done():
            pump_done = True
        time.sleep(interval)

    busy = False
    log.info("monitor_operations: busy set to False after completion")

# --- Unified background task ---
def action_task(request: ActionRequest):
    job_id = request.id
    pumpA = request.pumpA
    duration = request.time
      # Send motor command
    step_time = send_command_to_hardware(pumpA, duration)
    # Monitor operation
    monitor_operations(job_id, pumpA, step_time)

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

