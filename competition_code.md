```python
    def robot_controller(self):
        # Swarm Robotics Group B (sibh, frgm & otja)
        def get_mean_direction(bearings, opposite=False, ws=None):
            if not bearings:
                return None

            x, y = np.average(np.cos(bearings), weights=ws), np.average(np.sin(bearings), weights=ws)
            if opposite:
                return np.arctan2(-y, -x)
            return np.arctan2(y, x)

        def flock_controls():
            # weights for avoid, align, cohesion
            w_avoid = 0.8
            w_align = 1
            w_cohesion = 0.1
            w_light_cohesion = 2.5
            w_follow_light = 0.2
            speed = 1

            # trigger distance for avoidance behaviours
            min_flock_dist = 50

            flock_headings = [s['message']['heading'] for s in self.rab_signals]
            signal_bearings = [s['bearing'] for s in self.rab_signals]
            avoid_signal_bearings = [s['bearing'] for s in self.rab_signals if s['distance'] < min_flock_dist]
            light_bearing = [s['bearing'] for s in self.rab_signals if s['message']['comm_signal'] == 2]
            if light_bearing:
                w_cohesion = w_light_cohesion
            if self.light_intensity > 0.0:
                w_cohesion = w_light_cohesion
                self.set_rotation_and_speed(0, 0)
                return

            # calculate all relevant behaviour directions (is None if not relevant)
            avoid_dir = get_mean_direction(avoid_signal_bearings, opposite=True)
            align_dir = get_mean_direction(flock_headings)
            align_dir = self.get_relative_heading(align_dir) if align_dir is not None else align_dir
            cohesion_dir = get_mean_direction(signal_bearings)
            light_dir = get_mean_direction(light_bearing)

            # combine all relevant directions with weights
            weights = [w_avoid, w_align, w_cohesion, w_follow_light]
            control_dirs = [x for x in [avoid_dir, align_dir, cohesion_dir, light_dir] if x is not None]
            new_direction = get_mean_direction(control_dirs, ws=weights[:len(control_dirs)])

            # set new direction
            if new_direction is not None:
                self.set_rotation_and_speed(new_direction, MAX_SPEED * speed)
            else:
                self.set_rotation_and_speed(0, MAX_SPEED * speed)

        def disperse_controls():
            signal_bearings = [s['bearing'] for s in self.rab_signals]
            if not signal_bearings:
                self.set_rotation_and_speed(0, MAX_SPEED * 1)
                return

            x, y = np.mean(np.cos(signal_bearings)), np.mean(np.sin(signal_bearings))
            opposite_vector = np.arctan2(-y, -x)
            self.set_rotation_and_speed(opposite_vector, MAX_SPEED)

        # if close to any walls, immediatly steer away and return
        min_wall_dist = 50
        wall_bearings = [a for (r, a) in zip(self.prox_readings, self.prox_angles) if r['distance'] < min_wall_dist and (r['type'] == 'wall' or r['type'] == 'obstacle') ]
        if wall_bearings:
            opposite_wall_vector = get_mean_direction(wall_bearings, opposite=True)
            self.set_rotation_and_speed(opposite_wall_vector, MAX_SPEED)
            return


        light_found = [s for s in self.rab_signals if s['message']['comm_signal'] == 2 or s['message']['comm_signal'] == 1]
        if self.light_intensity > 0.0:
            self.broadcast_signal = 2
        elif light_found:
            self.broadcast_signal = 1
        else:
            self.broadcast_signal = 0

        if self.broadcast_signal == 0:
            disperse_controls()
        else:
            flock_controls()
```
