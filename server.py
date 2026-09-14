import math
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests
import uvicorn

app = FastAPI(title="AgriBot Full Visual Brain")
ESP32_MASTER_URL = "http://192.168.4.1"

STEPS_PER_REV = 1600.0           
SPOOL_RADIUS_M = 0.02            
M_PER_STEP = (2 * math.pi * SPOOL_RADIUS_M) / STEPS_PER_REV

abs_steps = np.zeros(4)
saved_corners = []

is_calibrated = False
POLES = np.zeros((4, 3))
WS_MIN = np.zeros(3)
WS_MAX = np.zeros(3)
current_ik_pos = np.zeros(3)
dim = [0,0,0]

ANCHORS_REL = np.array([
    [-0.1, -0.1, 0.05], [ 0.1, -0.1, 0.05],
    [ 0.1,  0.1, 0.05], [-0.1,  0.1, 0.05]
])

def compute_ik(pos):
    lengths = np.zeros(4)
    for i in range(4):
        lengths[i] = np.linalg.norm(POLES[i] - (pos + ANCHORS_REL[i]))
    return lengths

# --- VISUALIZER STATE ---
@app.get("/api/state")
def get_state():
    return {
        "calibrated": is_calibrated,
        "steps": abs_steps.tolist(),
        "corners": len(saved_corners),
        "pos": current_ik_pos.tolist(),
        "dim": dim
    }

# --- PHASE 1: INDIVIDUAL JOGGING ---
@app.get("/api/jog")
def jog_motor(motor: int, dir: int, steps: int = 400):
    global abs_steps
    payload = []
    for i in range(4):
        if i == motor:
            # Lowered speed to 600 to prevent NEMA 17 stall
            payload.append(f"{1 if dir > 0 else 0},{steps},600")
            abs_steps[i] += (dir * steps)
        else:
            payload.append("0,0,0")
            
    try:
        requests.get(f"{ESP32_MASTER_URL}/api/coord?data={'|'.join(payload)}", timeout=1)
        return {"status": "Success"}
    except:
        return {"status": "Hardware Unreachable"}

# --- PHASE 1 & 2: JOYSTICK ---
@app.get("/api/joystick")
def joystick_stream(vx: float, vy: float, vz: float):
    global current_ik_pos, abs_steps
    
    if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(vz) < 0.1:
        return {"status": "Stopped"}

    if not is_calibrated:
        # Raw drone-mixing for rough corner finding
        step_diff = np.zeros(4)
        step_diff[0] = ( vx + vy - vz) * 100  
        step_diff[1] = (-vx + vy - vz) * 100  
        step_diff[2] = (-vx - vy - vz) * 100  
        step_diff[3] = ( vx - vy - vz) * 100  
        step_diff = step_diff.astype(int)
    else:
        # True Decoupled 3D Kinematics
        target = current_ik_pos.copy()
        target[0] += vx * 0.02  # 2cm flat movement
        target[1] += vy * 0.02
        target[2] += vz * 0.02
        target = np.clip(target, WS_MIN, WS_MAX)
        
        old_len = compute_ik(current_ik_pos)
        new_len = compute_ik(target)
        step_diff = ((new_len - old_len) / M_PER_STEP).astype(int)
        current_ik_pos = target

    max_steps = np.max(np.abs(step_diff))
    if max_steps == 0: return {"status": "Limit"}

    # Lowered max speed for reliable torque
    speeds = (np.abs(step_diff) / max_steps) * 800.0

    payload = []
    for i in range(4):
        d_val = 1 if step_diff[i] >= 0 else 0
        payload.append(f"{d_val},{abs(step_diff[i])},{int(speeds[i])}")
        abs_steps[i] += step_diff[i]
        
    try:
        requests.get(f"{ESP32_MASTER_URL}/api/coord?data={'|'.join(payload)}", timeout=1)
        return {"status": "Moving"}
    except:
        return {"status": "Hardware Unreachable"}

# --- CALIBRATION ---
@app.get("/api/save_corner")
def save_corner():
    global saved_corners
    if len(saved_corners) < 8: saved_corners.append(abs_steps.copy())
    return {"count": len(saved_corners)}

@app.get("/api/finish_calibration")
def finish_calibration():
    global is_calibrated, POLES, WS_MIN, WS_MAX, current_ik_pos, dim
    if len(saved_corners) < 8: return {"status": "Need 8 corners"}

    corners_m = np.array(saved_corners) * M_PER_STEP
    min_cables = np.min(corners_m, axis=0)
    max_cables = np.max(corners_m, axis=0)

    dim[0] = (max_cables[0] - min_cables[0]) * 0.8  # Width X
    dim[1] = (max_cables[1] - min_cables[1]) * 0.8  # Length Y
    dim[2] = (max_cables[2] - min_cables[2]) * 0.8  # Height Z

    POLES = np.array([
        [0.0, 0.0, dim[2]], [dim[0], 0.0, dim[2]],
        [dim[0], dim[1], dim[2]], [0.0, dim[1], dim[2]],
    ])

    # Absolute safe limits
    WS_MIN = np.array([0.2, 0.2, 0.2])
    WS_MAX = np.array([dim[0]-0.2, dim[1]-0.2, dim[2]-0.2])
    current_ik_pos = np.array([dim[0]/2, dim[1]/2, dim[2]/2])
    
    is_calibrated = True
    return {"status": "Calibrated"}

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("index.html", "r") as f: return f.read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)