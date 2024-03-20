import pymunk

from config import config, CollisionTypes


class Wall(pymunk.Segment):
    def __init__(self, a, b, space):
        super().__init__(
            body=pymunk.Body(body_type=pymunk.Body.STATIC),
            a=a,
            b=b,
            radius=2
        )
        self.collision_type = CollisionTypes.WALL
        self.friction = config.physics.wall_friction
        space.add(self.body, self)
