from config import config
from preparticle import PreParticle


class Cloud:
    def __init__(self):
        self.x = config.screen.width // 2
        self.curr = PreParticle()
        self.next = PreParticle()

    def draw(self, screen, wait):
        self.curr.draw(screen, wait)
        self.next.pre_draw(screen)

    def release(self, space):
        return self.curr.release(space)

    def move_left(self, val=1):
        self.x = self.curr.set_x(self.x - val)

    def move_right(self, val=1):
        self.x = self.curr.set_x(self.x + val)

    def step(self):
        self.curr = self.next
        self.curr.set_x(self.x)
        self.next = PreParticle()
