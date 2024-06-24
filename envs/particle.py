import math


class Particle:
    INFINITY = math.inf
    EPS = 1e-10

    def __init__(self, index, rx, ry, vx, vy, radius, mass, kind, margin, friction=0.0):
        self.index = index

        assert rx - radius >= 0
        assert rx + radius <= 1
        assert ry - radius >= 0
        assert ry + radius <= 1
        self.rx = rx
        self.ry = ry
        self.vx = vx
        self.vy = vy

        assert kind in ('active', 'passive', 'goal')
        self.kind = kind

        self.radius = radius
        self.margin = margin
        self.mass = mass
        self.friction = friction
        self.count = 0

    @classmethod
    def create(cls, index, np_random, kind, off_wall_placement, velocity=0.01, friction=0.0):
        mass = 0.5
        radius = 1 / 14 * (math.sqrt(2) + 1) / 2
        margin = radius
        if off_wall_placement:
            margin += 2 * radius

        rx = np_random.random() * (1 - 2 * margin) + margin
        ry = np_random.random() * (1 - 2 * margin) + margin
        particle = cls(index, rx, ry, 0, 0, radius, mass, kind, margin, friction=friction)

        v = np_random.random() * velocity
        angle = np_random.random() * math.tau
        particle.set_velocity(v, angle)

        return particle

    def set_velocity(self, velocity, angle):
        self.vx = velocity * math.cos(angle)
        self.vy = velocity * math.sin(angle)

    def does_intersect_with_particle(self, particle):
        dx = particle.rx - self.rx
        dy = particle.ry - self.ry
        distance = math.sqrt(dx * dx + dy * dy)
        sigma = particle.radius + self.radius

        return distance - sigma <= self.EPS

    @staticmethod
    def _move(coordinate, velocity, acceleration, dt):
        coordinate += velocity * dt + 0.5 * acceleration * dt * dt
        velocity += acceleration * dt

        return coordinate, velocity

    def move(self, dt):
        v = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if v <= self.EPS:
            self.vx = self.vy = 0
            return

        t = dt
        if self.friction != 0:
            t = min(dt, v / self.friction)

        self.rx, self.vx = self._move(self.rx, self.vx, -self.vx / v * self.friction, t)
        self.ry, self.vy = self._move(self.ry, self.vy, -self.vy / v * self.friction, t)

    def is_stopped(self):
        return self.vx == 0 and self.vy == 0

    def time_to_hit(self, particle):
        if particle == self:
            return self.INFINITY

        dx = particle.rx - self.rx
        dy = particle.ry - self.ry
        dvx = particle.vx - self.vx
        dvy = particle.vy - self.vy
        dvdr = dx * dvx + dy * dvy
        if dvdr > 0:
            return self.INFINITY

        dvdv = dvx * dvx + dvy * dvy
        if dvdv == 0:
            return self.INFINITY

        drdr = dx * dx + dy * dy
        sigma = particle.radius + self.radius
        d = dvdr * dvdr - dvdv * (drdr - sigma * sigma)
        if d < 0:
            return self.INFINITY

        t = -(dvdr + math.sqrt(d)) / dvdv
        if t <= 0:
            return self.INFINITY

        return t

    def does_collide_with_particle(self, particle):
        dx = particle.rx - self.rx
        dy = particle.ry - self.ry
        distance = math.sqrt(dx * dx + dy * dy)
        sigma = particle.radius + self.radius
        if distance - sigma > self.EPS:
            return False

        dvx = particle.vx - self.vx
        dvy = particle.vy - self.vy
        dvdr = dx * dvx + dy * dvy
        if dvdr > 0:
            return False

        return True

    def time_to_hit_vertical_wall(self):
        if self.vx > 0:
            return (1 - self.rx - self.margin) / self.vx
        elif self.vx < 0:
            return (self.margin - self.rx) / self.vx
        else:
            return self.INFINITY

    def does_collide_with_vertical_wall(self):
        if self.vx > 0 and 1 - self.rx - self.margin < self.EPS:
            return True

        if self.vx < 0 and self.rx - self.margin < self.EPS:
            return True

        return False

    def time_to_hit_horizontal_wall(self):
        if self.vy > 0:
            return (1 - self.ry - self.margin) / self.vy
        elif self.vy < 0:
            return (self.margin - self.ry) / self.vy
        else:
            return self.INFINITY

    def does_collide_with_horizontal_wall(self):
        if self.vy > 0 and 1 - self.ry - self.margin < self.EPS:
            return True

        if self.vy < 0 and self.ry - self.margin < self.EPS:
            return True

        return False

    def bounce_off(self, particle):
        dx = particle.rx - self.rx
        dy = particle.ry - self.ry
        dvx = particle.vx - self.vx
        dvy = particle.vy - self.vy
        dvdr = dx * dvx + dy * dvy
        dist = self.radius + particle.radius

        magnitude = 2 * self.mass * particle.mass * dvdr / ((self.mass + particle.mass) * dist)
        if self.kind == 'goal':
            magnitude *= (self.mass + particle.mass) / self.mass
        elif particle.kind == 'goal':
            magnitude *= (self.mass + particle.mass) / particle.mass

        fx = magnitude * dx / dist
        fy = magnitude * dy / dist

        self.vx += int(self.kind != 'goal') * fx / self.mass
        self.vy += int(self.kind != 'goal') * fy / self.mass
        particle.vx -= int(particle.kind != 'goal') * fx / particle.mass
        particle.vy -= int(particle.kind != 'goal') * fy / particle.mass

        self.count += 1
        particle.count += 1

    def bounce_off_vertical_wall(self):
        self.vx = -self.vx
        self.count += 1

    def bounce_off_horizontal_wall(self):
        self.vy = -self.vy
        self.count += 1

    def kinetic_energy(self):
        return 0.5 * self.mass * (self.vx * self.vx + self.vy * self.vy)
