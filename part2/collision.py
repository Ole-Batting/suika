import numpy as np

from config import config
from particle import Particle


def resolve_collision(particle1, particle2, space):
    distance = np.linalg.norm(particle1.pos - particle2.pos)
    if distance < 2 * particle1.radius:
        particle1.kill(space)
        particle2.kill(space)
        new_particle = Particle(
            np.mean([particle1.pos, particle2.pos], axis=0),
            particle1.n + 1,
            space,
        )
        for p in space.shapes:
            if isinstance(p, Particle) and p.alive:
                vector = p.pos - new_particle.pos
                distance = np.linalg.norm(vector)
                if distance < new_particle.radius + p.radius:
                    impulse = config.physics.impulse * vector / (distance ** 2)
                    p.body.apply_impulse_at_local_point(tuple(impulse))


def collide(arbiter, space, data):
    particle1, particle2 = arbiter.shapes
    alive = particle1.alive and particle2.alive
    same = particle1.n == particle2.n
    particle1.has_collided = not same
    particle2.has_collided = not same
    if same and alive:
        resolve_collision(particle1, particle2, space)
        data["score"] += config[particle1.n, "points"]
    return not same and alive
