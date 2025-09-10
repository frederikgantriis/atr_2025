import pygame
import numpy as np

# Pygame setup
WIDTH, HEIGHT = 800, 600
BG_COLOR = (30, 30, 30)
ROBOT_COLOR = (200, 255, 255)
OBSTACLE_COLOR = (200, 50, 50)
FONT_COLOR = (255, 255, 255)

SIM_DT = 1 / 60.0

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "Robots Detecting Each Other and Obstacles with Proximity & RAB")
font = pygame.font.SysFont(None, 20)

# Surrounding walls
ARENA_BOUNDS = {
    'left': 0,
    'right': WIDTH,
    'top': 0,
    'bottom': HEIGHT
}

# Parameters
NUM_ROBOTS = 2
ROBOT_RADIUS = 10

NUM_PROX_SENSORS = 6
NUM_RAB_SENSORS = 12

PROX_SENSOR_RANGE = 60  # pixels
RAB_RANGE = 150  # pixels

MAX_SPEED = 50
# radians/sec - a real robot like the e-puck or TurtleBot typically turns at 90–180 deg/sec (≈ 1.5–3.1 rad/sec)
MAX_TURN = 3

# sensor noise and dropout
# std dev of directional noise to bearing in RAB:  0.1 rad. = ~5.7 degree in the bearing
RAB_NOISE_BEARING = 0
RAB_DROPOUT = 0  # chance to drop a signal
LIGHT_NOISE_STD = 0  # noise in perceived light
ORIENTATION_NOISE_STD = 0  # noise in IMU readings of the robot’s own orientation

# noise in the motion model (simulates actuation/motor errors)
MOTION_NOISE_STD = 0  # Try 0.5   # Positional noise in dx/dy (pixels)
HEADING_NOISE_STD = 0  # Try 0.01 # Rotational noise in heading (radians)


def rotate_vector(vec, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*vec[0] - s*vec[1], s*vec[0] + c*vec[1]])


class LightSource:
    def __init__(self, pos, intensity=1.0, core_radius=0, decay_radius=200):
        self.pos = np.array(pos, dtype=float)
        self.intensity = intensity
        self.core_radius = core_radius
        self.decay_radius = decay_radius

    def get_intensity_at(self, dist):
        if dist > self.decay_radius:
            return 0.0
        if dist < self.core_radius:
            return self.intensity
        # Inverse-square decay beyond the center radius (you can change to linear or exponential)
        return self.intensity * max(0.0, 1.0 - ((dist - self.core_radius) / (self.decay_radius - self.core_radius)) ** 2)

# Utility function to sum light intensity from all sources


def _get_light_intensity(pos):
    raw_intensity = sum(source.get_intensity_at(
        np.linalg.norm(pos - source.pos)) for source in LIGHT_SOURCES)
    noise_factor = np.random.normal(1.0, LIGHT_NOISE_STD)
    return np.clip(raw_intensity * noise_factor, 0.0, 1.0)


class Obstacle:
    def __init__(self, pos, radius):
        self.pos = np.array(pos, dtype=float)
        self.radius = radius
        self.type = "obstacle"  # used in sensing


OBSTACLES = [
    # Obstacle(pos=(200, 150), radius=20),
    # Obstacle(pos=(600, 120), radius=30),
]

LIGHT_SOURCES = [
    # LightSource(pos=(100, 100), intensity=1.0, core_radius=50, decay_radius=300),
    # LightSource(pos=(700, 500), intensity=0.9, core_radius=10, decay_radius=100)
]


class Robot:
    def __init__(self, id, pos, heading):
        self.id = id
        self._pos = np.array(pos, dtype=float)
        self._heading = heading
        self._radius = ROBOT_RADIUS
        self._linear_velocity = MAX_SPEED * 0.5
        self._angular_velocity = 0

        # signal broadcast via RAB (a very short message)
        self.broadcast_signal = None

        # Sensor readings
        # proximity sensors
        self.prox_angles = np.pi / NUM_PROX_SENSORS + \
            np.linspace(0, 2 * np.pi, NUM_PROX_SENSORS, endpoint=False)
        self.prox_readings = [
            {"distance": PROX_SENSOR_RANGE, "type": None}
            for _ in range(NUM_PROX_SENSORS)
        ]
        # RAB sensor
        self.rab_angles = np.pi / NUM_RAB_SENSORS + \
            np.linspace(0, 2 * np.pi, NUM_RAB_SENSORS, endpoint=False)
        self.rab_signals = []
        # light sensor
        self.light_intensity = 0.0
        # IMU (Inertial Measurement Unit) sensor providing robot's orientation
        self.orientation = 0.0

    def move(self, dt):
        # Update heading
        self._heading += self._angular_velocity * dt
        self._heading += np.random.normal(0, HEADING_NOISE_STD)
        self._heading %= 2 * np.pi  # keep in [0, 2π)

        # Update position
        dx = self._linear_velocity * np.cos(self._heading) * dt
        dy = self._linear_velocity * np.sin(self._heading) * dt
        dx += np.random.normal(0, MOTION_NOISE_STD)
        dy += np.random.normal(0, MOTION_NOISE_STD)
        self._pos += np.array([dx, dy])

        # Arena bounds clipping
        self._pos[0] = np.clip(
            self._pos[0], self._radius, WIDTH - self._radius)
        self._pos[1] = np.clip(
            self._pos[1], self._radius, HEIGHT - self._radius)

    def compute_distance_to_wall(self, direction, bounds, max_range):
        x, y = self._pos
        dx, dy = direction
        distances = []
        if dx != 0:
            if dx > 0:
                t = (bounds['right'] - x) / dx
            else:
                t = (bounds['left'] - x) / dx
            if t > 0:
                distances.append(t)
        if dy != 0:
            if dy > 0:
                t = (bounds['bottom'] - y) / dy
            else:
                t = (bounds['top'] - y) / dy
            if t > 0:
                distances.append(t)
        wall_distance = min(distances) if distances else max_range
        wall_distance = max(wall_distance, 0)
        return min(wall_distance, max_range)

    def read_sensors(self, robots, obstacles, arena_bounds):

        # Empty the sensors
        self.prox_readings = [
            {"distance": PROX_SENSOR_RANGE, "type": None}
            for _ in range(NUM_PROX_SENSORS)
        ]
        self.rab_signals = []

        # Light sensing
        self.light_intensity = _get_light_intensity(self._pos)

        # Detect other robots
        for other in robots:
            if other.id == self.id:
                continue

            rel_vec = other._pos - self._pos
            distance = max(0, np.linalg.norm(rel_vec) - other._radius)
            if distance > max(PROX_SENSOR_RANGE, RAB_RANGE):
                continue

            bearing = (np.arctan2(
                rel_vec[1], rel_vec[0]) - self._heading) % (2 * np.pi)

            # Communication (RAB)
            if distance <= RAB_RANGE:
                # signal dropout
                dropout_probability = RAB_DROPOUT  # chance to drop a signal

                if np.random.rand() > dropout_probability:
                    # adding noise (directional error) to bearing
                    bearing = (bearing + np.random.normal(0,
                               RAB_NOISE_BEARING)) % (2 * np.pi)

                    rab_idx = int((bearing / (2 * np.pi)) * NUM_RAB_SENSORS)

                    self.rab_signals.append({
                        'message': {'heading': other.orientation, 'comm_signal': other.broadcast_signal},
                        'distance': distance,
                        'bearing': self.rab_angles[rab_idx],  # local
                        'sensor_idx': rab_idx,
                        'intensity': 1 / ((distance / RAB_RANGE) ** 2 + 1),
                    })

            # Also treat robot as obstacle (for IR)
            if distance <= PROX_SENSOR_RANGE:
                prox_idx = int((bearing / (2 * np.pi)) * NUM_PROX_SENSORS)
                if distance < self.prox_readings[prox_idx]["distance"]:
                    self.prox_readings[prox_idx] = {
                        "distance": distance, "type": "robot"}

        # Detect obstacles
        for obs in obstacles:
            rel_vec = obs.pos - self._pos
            distance = max(0, np.linalg.norm(rel_vec) - obs.radius)
            if distance <= PROX_SENSOR_RANGE:
                bearing = (np.arctan2(
                    rel_vec[1], rel_vec[0]) - self._heading) % (2 * np.pi)
                prox_idx = int((bearing / (2 * np.pi)) * NUM_PROX_SENSORS)
                if distance < self.prox_readings[prox_idx]["distance"]:
                    self.prox_readings[prox_idx] = {
                        "distance": distance, "type": "obstacle"}

        # Wall sensing (raycast style)
        for i, angle in enumerate(self.prox_angles):
            global_angle = (self._heading + angle) % (2 * np.pi)
            direction = np.array([np.cos(global_angle), np.sin(global_angle)])
            wall_dist = self.compute_distance_to_wall(
                direction, arena_bounds, PROX_SENSOR_RANGE)
            if wall_dist < self.prox_readings[i]["distance"]:
                self.prox_readings[i] = {"distance": wall_dist, "type": "wall"}

        # Read IMU for own orientation
        self.orientation = (
            self._heading + np.random.normal(0, ORIENTATION_NOISE_STD)) % (2 * np.pi)

    def _set_velocity(self, linear, angular):
        # Internal use only. Use set_rotation_and_speed instead
        assert 0 <= linear <= MAX_SPEED, "Linear velocity out of bounds"
        assert -MAX_TURN <= angular <= MAX_TURN, "Angular velocity out of bounds"
        self._linear_velocity = linear
        self._angular_velocity = angular

    def compute_angle_diff(self, target_angle):
        # Returns shortest signed angle between current heading and target
        return (target_angle - self._heading + np.pi) % (2 * np.pi) - np.pi

    def get_relative_heading(self, neighbor_heading):
        # Convert a neighbor's global heading into this robot's local frame (radians).
        #     Positive = CCW, Negative = CW.
        return (neighbor_heading - self.orientation + np.pi) % (2 * np.pi) - np.pi

    def set_rotation_and_speed(self, delta_bearing, target_speed, kp=0.5):
        """
        Sets angular and linear velocity using a proportional controller
        to achieve the given relative turn (rotation) with the given target speed.
        Robot-frame API: delta_bearing is relative to current heading (rad)
        + = turn left (CCW), - = turn right (CW).
        """
        target_heading = (self._heading + delta_bearing) % (2 * np.pi)
        angle_diff = self.compute_angle_diff(target_heading)
        angular_velocity = np.clip(kp * angle_diff, -MAX_TURN, MAX_TURN)
        target_speed = np.clip(target_speed, 0, MAX_SPEED)
        # Slow down when turning sharply
        linear_velocity = target_speed * \
            (1 - min(abs(angle_diff) / np.pi, 1)) * 0.9 + 0.1
        self._set_velocity(linear_velocity, angular_velocity)

    def controller_init(self):
        pass

    def robot_controller(self):
        """
            Implement your control logic here.
            You can access:
            - self.rab_signals: list of received messages from other robots
            - self.prox_readings: proximity sensor data
            - self.light_intensity: light at current location

            Use only self.set_rotation_and_speed(...) to move the robot.

            DO NOT modify robot._linear_velocity or robot._angular_velocity directly. DO NOT modify move()
            """
        # Example: move forward
        self.set_rotation_and_speed(0, MAX_SPEED * 0.5)

    def draw(self, screen):
        # --- IR proximity sensors ---
        for i, reading in enumerate(self.prox_readings):
            dist = reading["distance"]
            obj_type = reading["type"]

            angle = self._heading + self.prox_angles[i]
            sensor_dir = np.array([np.cos(angle), np.sin(angle)])
            end_pos = self._pos + sensor_dir * dist

            # Color code by detected object type
            if obj_type == "robot":
                color = (0, 150, 255)  # Blue
            elif obj_type == "obstacle":
                color = (255, 165, 0)  # Orange
            elif obj_type == "wall":
                color = (255, 255, 100)  # Yellow
            else:
                color = (20, 80, 20)  # Green (no hit)

            pygame.draw.line(screen, color, self._pos, end_pos, 2)
            pygame.draw.circle(screen, color, end_pos.astype(int), 3)

        # --- RAB signals ---
        for sig in self.rab_signals:
            sig_angle = self._heading + self.rab_angles[sig['sensor_idx']]
            sensor_dir = np.array([np.cos(sig_angle), np.sin(sig_angle)])

            start = self._pos + sensor_dir * (self._radius + 3)
            end = self._pos + sensor_dir * (self._radius + 3 + sig['distance'])

            intensity_color = 55+int(200 * (sig['intensity']*2-1))
            color = (intensity_color, 50, intensity_color)

            pygame.draw.line(screen, color, start, end, 2)

        # --- Robot body ---
        pygame.draw.circle(screen, ROBOT_COLOR,
                           self._pos.astype(int), self._radius)

        # --- Heading indicator ---
        heading_vec = rotate_vector(
            np.array([self._radius + 2, 0]), self._heading)
        pygame.draw.line(screen, ROBOT_COLOR, self._pos,
                         self._pos + heading_vec, 3)


def draw_obstacles(screen):
    for obs in OBSTACLES:
        pygame.draw.circle(screen, (120, 120, 120),
                           obs.pos.astype(int), obs.radius)


def draw_light_sources(screen):
    for light in LIGHT_SOURCES:
        # Draw fading light circle
        for r in range(light.decay_radius, light.core_radius, -10):
            alpha = int(255 * light.get_intensity_at(r))
            surface = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (light.intensity*255,
                               light.intensity*235, 0, alpha), (r, r), r)
            screen.blit(surface, (light.pos[0] - r, light.pos[1] - r))

        # Draw core of the light
        pygame.draw.circle(screen, (light.intensity*255, light.intensity*255, 0),
                           light.pos.astype(int), max(5, light.core_radius))


def logging_init():  # initialize your log file
    pass


def log_metrics(frame_count, total_time, metrics):  # write to your log file
    pass


def logging_close():  # close your log file
    pass


def compute_metrics():  # pass as many arguments as you need and compute relevant metrics to be logged for performance analysis
    return []


def main():
    clock = pygame.time.Clock()
    dt = SIM_DT
    robots = []

    np.random.seed(42)
    for i in range(NUM_ROBOTS):
        pos = np.random.uniform([ROBOT_RADIUS, ROBOT_RADIUS], [
                                WIDTH - ROBOT_RADIUS, HEIGHT - ROBOT_RADIUS])
        heading = np.random.uniform(0, 2 * np.pi)
        robots.append(Robot(i, pos, heading))

    # initialize robot controllers
    for robot in robots:
        robot.controller_init()

    logging_init()

    frame_count = 0
    total_time = 0.0
    running = True
    paused = False
    visualize = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif event.key == pygame.K_SPACE:
                    visualize = not visualize
                    print(
                        "Visualization", "enabled" if visualize else "disabled", "at", total_time)

        if not paused:
            total_time += dt  # accumulate time

            for robot in robots:
                robot.read_sensors(robots, OBSTACLES, ARENA_BOUNDS)

            for robot in robots:
                robot.robot_controller()

            for robot in robots:
                robot.move(dt)

            metrics = compute_metrics()
            log_metrics(frame_count, total_time, metrics)

            frame_count += 1

        if visualize:
            clock.tick(60 if not paused else 10)
            screen.fill(BG_COLOR)
            draw_light_sources(screen)
            draw_obstacles(screen)
            for robot in robots:
                robot.draw(screen)
            if paused:
                txt = font.render("PAUSED", True, (255, 100, 100))
                screen.blit(txt, (10, 10))
            pygame.display.flip()
            pygame.display.set_caption("Robot Sim — VISUAL MODE")
        else:
            pygame.display.set_caption(
                "Robot Sim — PAUSED in HEADLESS" if paused else "Robot Sim — HEADLESS")

    pygame.quit()
    logging_close()


if __name__ == "__main__":
    main()
