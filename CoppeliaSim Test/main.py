from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

# =========================================================
# 1. CONNECT TO COPPELIASIM
# =========================================================

print("Connecting to CoppeliaSim...")

client = RemoteAPIClient()
sim = client.require("sim")

print("Connected to CoppeliaSim!")


# =========================================================
# 2. GET SENSOR HANDLES
# =========================================================

left_sensor = sim.getObject("/youBot/left_sensor")
center_sensor = sim.getObject("/youBot/center_sensor")
right_sensor = sim.getObject("/youBot/right_sensor")

print("Sensor handles loaded successfully!")


# =========================================================
# 3. GET YOUBOT SCRIPT
# =========================================================

robot_script = sim.getObject("/youBot/Script")

print("YouBot script loaded successfully!")


# =========================================================
# 4. READ SENSOR DISTANCE
# =========================================================

def get_distance(sensor):

    detected, distance, point, obj_handle, normal = sim.readProximitySensor(sensor)

    if detected:
        return distance

    # No obstacle detected
    return 2.0


# =========================================================
# 5. MOVE ROBOT
# =========================================================

def move_robot(forward, sideways, rotation):

    sim.callScriptFunction(
        "moveRobot",
        robot_script,
        forward,
        sideways,
        rotation
    )


# =========================================================
# 6. OBSTACLE AVOIDANCE
# =========================================================

SAFE_DISTANCE = 0.4

FORWARD_SPEED = 0.2
TURN_SPEED = 0.25
REVERSE_SPEED = -0.1


print("\n================ OBSTACLE AVOIDANCE STARTED ================")


while True:

    # ---------------------------------------------------------
    # READ THREE SENSORS
    # ---------------------------------------------------------

    left_distance = get_distance(left_sensor)
    center_distance = get_distance(center_sensor)
    right_distance = get_distance(right_sensor)


    # ---------------------------------------------------------
    # DISPLAY SENSOR VALUES
    # ---------------------------------------------------------

    print(
        f"L: {left_distance:.2f} m | "
        f"C: {center_distance:.2f} m | "
        f"R: {right_distance:.2f} m"
    )


    # =========================================================
    # CASE 1: FRONT IS CLEAR
    # =========================================================

    if center_distance > SAFE_DISTANCE:

        print("ACTION: FORWARD")

        move_robot(
            FORWARD_SPEED,
            0,
            0
        )


    # =========================================================
    # CASE 2: FRONT BLOCKED
    # =========================================================

    else:

        print("OBSTACLE AHEAD!")


        # -----------------------------------------------------
        # LEFT SIDE IS MORE OPEN
        # -----------------------------------------------------

        if left_distance > right_distance:

            print("ACTION: TURN LEFT")

            move_robot(
                0,
                0,
                TURN_SPEED
            )
            time.sleep(1.5)


        # -----------------------------------------------------
        # RIGHT SIDE IS MORE OPEN
        # -----------------------------------------------------

        elif right_distance > left_distance:

            print("ACTION: TURN RIGHT")

            move_robot(
                0,
                0,
                -TURN_SPEED
            )
            time.sleep(1.5)


        # -----------------------------------------------------
        # BOTH SIDES ARE BLOCKED
        # -----------------------------------------------------

        else:

            print("ACTION: REVERSE")

            move_robot(
                REVERSE_SPEED,
                0,
                0
            )


    # ---------------------------------------------------------
    # SMALL DELAY
    # ---------------------------------------------------------

    time.sleep(0.1)