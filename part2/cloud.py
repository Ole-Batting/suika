from preparticle import PreParticle


class Cloud:
    def __init__(self):
        self.curr = PreParticle()
        self.next = PreParticle()

    def draw(self, screen, wait):
        self.curr.draw(screen, wait)
        self.next.pre_draw(screen)

    def release(self, space):
        return self.curr.release(space)

    def step(self):
        self.curr = self.next
        self.next = PreParticle()
