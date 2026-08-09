from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import pickle
import warnings

# Naya: Scikit-learn ki extra warnings chupane ke liye
warnings.filterwarnings("ignore") 

# ==========================================
# 1. CONNECT & GET HANDLES
# ==========================================
client = RemoteAPIClient()
sim = client.require("sim")

front_left = sim.getObject("/youBot/rollingJoint_fl")
front_right = sim.getObject("/youBot/rollingJoint_fr")
rear_left = sim.getObject("/youBot/rollingJoint_rl")
rear_right = sim.getObject("/youBot/rollingJoint_rr")

left_sensor = sim.getObject("/youBot/left_sensor")
center_sensor = sim.getObject("/youBot/center_sensor")
right_sensor = sim.getObject("/youBot/right_sensor")

# ==========================================
# 2. MOVEMENT FUNCTIONS
# ==========================================
def move_forward(speed=0.2):
    sim.setJointTargetVelocity(front_left, speed)
    sim.setJointTargetVelocity(front_right, speed)
    sim.setJointTargetVelocity(rear_left, speed)
    sim.setJointTargetVelocity(rear_right, speed)

def move_backward(speed=0.2):
    sim.setJointTargetVelocity(front_left, -speed)
    sim.setJointTargetVelocity(front_right, -speed)
    sim.setJointTargetVelocity(rear_left, -speed)
    sim.setJointTargetVelocity(rear_right, -speed)

def turn_right(speed=0.4):
    sim.setJointTargetVelocity(front_left, speed)
    sim.setJointTargetVelocity(rear_left, speed)
    sim.setJointTargetVelocity(front_right, -speed)
    sim.setJointTargetVelocity(rear_right, -speed)

def turn_left(speed=0.4):
    sim.setJointTargetVelocity(front_left, -speed)
    sim.setJointTargetVelocity(rear_left, -speed)
    sim.setJointTargetVelocity(front_right, speed)
    sim.setJointTargetVelocity(rear_right, speed)

def stop_robot():
    sim.setJointTargetVelocity(front_left, 0)
    sim.setJointTargetVelocity(front_right, 0)
    sim.setJointTargetVelocity(rear_left, 0)
    sim.setJointTargetVelocity(rear_right, 0)

def get_distance(sensor_handle):
    res = sim.readProximitySensor(sensor_handle)
    return round(res[1], 4) if res[0] == 1 else 2.0 

# ==========================================
# 3. LOAD THE AI BRAIN
# ==========================================
print("Loading AI Model (smart_brain.pkl)...")
with open('smart_brain.pkl', 'rb') as file:
    ai_model = pickle.load(file)
print("Brain Loaded Successfully!")

# ==========================================
# 4. MAIN CONTROL LOOP (AI DRIVEN)
# ==========================================
# Safe Auto-Start Logic
sim.stopSimulation()
while sim.getSimulationState() != sim.simulation_stopped:
    time.sleep(0.1)
    
sim.startSimulation()
while sim.getSimulationState() == sim.simulation_stopped:
    time.sleep(0.1)
    
print("Simulation Started! Robot is now driven by AI.")

start_time = time.time()

# Run for 30 seconds
while (time.time() - start_time) < 3000:
    
    # 1. Sense the Environment
    dL = get_distance(left_sensor)
    dC = get_distance(center_sensor)
    dR = get_distance(right_sensor)
    
    # 2. Ask the AI Model what to do (Prediction)
    # Humein features 2D array me dene hote hain, isliye double brackets [[ ]]
    predicted_action = ai_model.predict([[dL, dC, dR]])[0]
    
    # 3. Execute the Action
    if predicted_action == 0:
        move_forward(0.2)
        action_name = "FORWARD"
    elif predicted_action == 1:
        turn_right(0.4)
        action_name = "RIGHT"
    elif predicted_action == 2:
        turn_left(0.4)
        action_name = "LEFT"
    elif predicted_action == 3:
        move_backward(0.2)
        action_name = "REVERSE"
        
    print(f"Sensors [L:{dL} C:{dC} R:{dR}] ==> AI Decided: {action_name}")
        
    time.sleep(0.05) 

# ==========================================
# 5. CLEANUP
# ==========================================
stop_robot()
sim.stopSimulation()
print("Simulation Stopped! Run complete.")