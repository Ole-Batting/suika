# !!animate
from _6 import *  # !!ignore
# Collision Handler
handler = space.add_collision_handler(1, 1)


def collide(arbiter, space, data):
    sh1, sh2 = arbiter.shapes
    _mapper = data["mapper"]
    pa1 = _mapper[sh1]
    pa2 = _mapper[sh2]
    cond = bool(pa1.n != pa2.n)
    pa1.has_collided = cond
    pa2.has_collided = cond
    if not cond:
        new_particle = resolve_collision(pa1, pa2, space, data["particles"], _mapper)
        data["particles"].append(new_particle)
        data["score"] += POINTS[pa1.n]
    return cond


handler.begin = collide
handler.data["mapper"] = shape_to_particle
handler.data["particles"] = particles
handler.data["score"] = 0
