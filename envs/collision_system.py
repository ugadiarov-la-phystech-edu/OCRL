import heapq
from typing import List

import numpy as np
import skimage
from matplotlib import pyplot as plt

from envs.particle import Particle


# https://algs4.cs.princeton.edu/code/
# https://algs4.cs.princeton.edu/code/edu/princeton/cs/algs4/CollisionSystem.java.html


class Event:
    def __init__(self, a: Particle, b: Particle, time: float):
        self.a = a
        self.b = b
        self.time = time
        self.count_a = a.count if a is not None else None
        self.count_b = b.count if b is not None else None

    def __lt__(self, other):
        return self.time < other.time

    def is_valid(self):
        if self.a is not None and self.a.count != self.count_a:
            return False
        if self.b is not None and self.b.count != self.count_b:
            return False

        return True


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        color = 255 * np.asarray(cm(1. * i / num_colors), dtype=np.float32)[:3]
        colors.append(color.astype(np.uint8))

    return colors


def diamond(width, im_size):
    rr, cc = [-0.5, width / 2 - 0.5, width, width / 2 - 0.5], [width / 2 - 0.5, -0.5, width / 2 - 0.5, width]
    return skimage.draw.polygon(rr, cc, None)


def square(width, im_size):
    rr, cc = [0, width - 1, width - 1, 0], [0, 0, width - 1, width - 1]
    return skimage.draw.polygon(rr, cc, None)


def triangle(width, im_size):
    rr, cc = [width - 1, width - 1, -1.5], [0, width - 1, width / 2 - 0.5]
    return skimage.draw.polygon(rr, cc, None)


def circle(width, im_size):
    radius = width / 2
    return skimage.draw.ellipse(radius - 0.5, radius - 0.5, radius, radius, None)


def cross(width, im_size):
    diff1 = width / 3
    diff2 = 2 * width / 3 - 1
    rr = [diff1, diff2, diff2, width - 1, width - 1,
          diff2, diff2, diff1, diff1, 0, 0, diff1]
    cc = [0, 0, diff1, diff1, diff2, diff2, width - 1,
          width - 1, diff2, diff2, diff1, diff1]
    return skimage.draw.polygon(rr, cc, None)


def pentagon(width, im_size):
    diff1 = width / 3 - 1
    diff2 = 2 * width / 3
    rr = [width / 2 - 0.5, width - 1, width - 1, width / 2 - 0.5, -1.5]
    cc = [-0.5, diff1, diff2, width - 0.5, width / 2 - 0.5]
    return skimage.draw.polygon(rr, cc, None)


def parallelogram(width, im_size):
    rr, cc = [0, width - 1, width - 1, 0], [0, width / 2 - 0.5, width - 1, width / 2 - 0.5]
    return skimage.draw.polygon(rr, cc, None)


def scalene_triangle(width, im_size):
    rr, cc = [-0.5, width - 0.5, width / 2 - 0.5], [width / 2 - 0.5, -0.5, width - 0.5]
    return skimage.draw.polygon(rr, cc, None)


def render_shape(idx, scale):
    shape_id = idx % 8
    if shape_id == 0:
        rr, cc = circle(scale, None)
    elif shape_id == 1:
        rr, cc = triangle(
            scale, None)
    elif shape_id == 2:
        rr, cc = square(
            scale, None)
    elif shape_id == 3:
        rr, cc = parallelogram(
            scale, None)
    elif shape_id == 4:
        rr, cc = cross(
            scale, None)
    elif shape_id == 5:
        rr, cc = diamond(
            scale, None)
    elif shape_id == 6:
        rr, cc = pentagon(
            scale, None)
    else:
        rr, cc = scalene_triangle(
            scale, None)

    return rr, cc


class CollisionSystem:
    HZ = 0.5
    ACTIVE_ACTIVE = 'active_active'
    ACTIVE_GOAL = 'active_goal'
    ACTIVE_PASSIVE = 'active_passive'
    ACTIVE_WALL = 'active_wall'
    PASSIVE_GOAL = 'passive_goal'
    PASSIVE_PASSIVE = 'passive_passive'
    PASSIVE_WALL = 'passive_wall'

    def __init__(self, n_particles, size=70, simulation_limit=10000, active_disappear_on_hit_goal=False,
                 active_disappear_on_hit_wall=False, passive_disappear_on_hit_wall=False, visualize=False,
                 frame_rate=25):
        self.particles = None
        self.n_particles = n_particles
        self.events = None
        self.t = 0
        self.colors = get_colors()
        self.size = size
        self.simulation_limit = simulation_limit
        self.frame_rate = frame_rate
        self.scale = self.size // 7
        self.shapes = [render_shape(idx, self.scale) for idx in range(self.n_particles)]
        self.active_disappear_on_hit_goal = active_disappear_on_hit_goal
        self.active_disappear_on_hit_wall = active_disappear_on_hit_wall
        self.passive_disappear_on_hit_wall = passive_disappear_on_hit_wall
        self.visualize = visualize
        self.fig = None
        self.ax = None

    def reset(self):
        self.events = None
        self.t = 0
        if self.visualize and self.ax is not None:
            self.ax.clear()

    def set_particles(self, particles: List[Particle]):
        assert len(particles) == self.n_particles, f'len(particles)={len(particles)}, expected: {self.n_particles}'
        self.particles = particles
        if self.visualize:
            if self.fig is not None:
                self.ax.clear()
                self.fig.clear()
                plt.close(self.fig)

            self.fig, self.ax = plt.subplots(1, 1)

    def _predict(self, particle: Particle, limit):
        if particle is None:
            return

        for other_particle in self.particles:
            if other_particle is None:
                continue

            dt = particle.time_to_hit(other_particle)
            if self.t + dt <= limit:
                heapq.heappush(self.events, Event(particle, other_particle, self.t + dt))

        dt_x = particle.time_to_hit_vertical_wall()
        dt_y = particle.time_to_hit_horizontal_wall()
        if self.t + dt_x <= limit:
            heapq.heappush(self.events, Event(particle, None, self.t + dt_x))

        if self.t + dt_y <= limit:
            heapq.heappush(self.events, Event(None, particle, self.t + dt_y))

    def get_observation(self):
        image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for particle, shape, color in zip(self.particles, self.shapes, self.colors):
            if particle is None:
                continue

            rr = shape[0] + round(self.size * particle.rx - self.scale / 2)
            cc = shape[1] + round(self.size * particle.ry - self.scale / 2)
            image[rr, cc, :] = color

        return image

    def _redraw(self, limit):
        image = self.get_observation()
        self.ax.clear()
        self.ax.imshow(image)
        plt.show(block=False)
        plt.pause(1 / self.frame_rate)
        if self.t < limit:
            heapq.heappush(self.events, Event(None, None, self.t + 1 / self.HZ))

    @classmethod
    def get_interaction_dict(cls):
        return {cls.ACTIVE_ACTIVE: 0, cls.ACTIVE_GOAL: 0, cls.ACTIVE_PASSIVE: 0, cls.ACTIVE_WALL: 0, cls.PASSIVE_GOAL:0,
                cls.PASSIVE_PASSIVE: 0, cls.PASSIVE_WALL: 0}

    def simulate(self):
        self.events = []
        interaction_dict = self.get_interaction_dict()
        for particle in self.particles:
            if particle is None:
                continue

            self._predict(particle, self.simulation_limit)

        if self.visualize:
            heapq.heappush(self.events, Event(None, None, 0))

        while len(self.events) > 0:
            if all(particle.is_stopped() for particle in self.particles if particle is not None):
                break

            event = heapq.heappop(self.events)
            if not event.is_valid():
                continue

            particle_a = event.a
            particle_b = event.b
            for particle in self.particles:
                if particle is None:
                    continue

                particle.move(event.time - self.t)

            self.t = event.time
            if particle_a is not None and particle_b is not None and particle_a.does_collide_with_particle(particle_b):
                particle_a.bounce_off(particle_b)
                if particle_a.kind == 'goal' or particle_b.kind == 'goal':
                    if particle_b.kind == 'goal':
                        tmp = particle_a
                        particle_a = particle_b
                        particle_b = tmp

                    if particle_b.kind == 'passive':
                        interaction_dict[self.PASSIVE_GOAL] += 1
                        self.particles[particle_b.index] = None
                    elif particle_b.kind == 'active':
                        interaction_dict[self.ACTIVE_GOAL] += 1
                        if self.active_disappear_on_hit_goal:
                            self.particles[particle_b.index] = None
                elif particle_a.kind == particle_b.kind:
                    if particle_a.kind == 'active':
                        interaction_dict[self.ACTIVE_ACTIVE] += 1
                    elif particle_a.kind == 'passive':
                        interaction_dict[self.PASSIVE_PASSIVE] += 1
                else:
                    interaction_dict[self.ACTIVE_PASSIVE] += 1
            elif particle_a is not None and particle_b is None and particle_a.does_collide_with_vertical_wall():
                particle_a.bounce_off_vertical_wall()
                interaction_dict[f'{particle_a.kind}_wall'] += 1
                if particle_a.kind == 'passive' and self.passive_disappear_on_hit_wall:
                    self.particles[particle_a.index] = None
                elif particle_a.kind == 'active' and self.active_disappear_on_hit_wall:
                    self.particles[particle_a.index] = None
            elif particle_a is None and particle_b is not None and particle_b.does_collide_with_horizontal_wall():
                particle_b.bounce_off_horizontal_wall()
                interaction_dict[f'{particle_b.kind}_wall'] += 1
                if particle_b.kind == 'passive' and self.passive_disappear_on_hit_wall:
                    self.particles[particle_b.index] = None
                elif particle_b.kind == 'active' and self.active_disappear_on_hit_wall:
                    self.particles[particle_b.index] = None
            elif particle_a is None and particle_b is None:
                self._redraw(self.simulation_limit)

            self._predict(particle_a, self.simulation_limit)
            self._predict(particle_b, self.simulation_limit)

        return interaction_dict
