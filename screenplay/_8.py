# !!animate
from _7 import *  # !!ignore
# Main game loop
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_RETURN, pygame.K_SPACE]:
                particles.append(next_particle.release(space, shape_to_particle))
                wait_for_next = NEXT_DELAY
            elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                pygame.quit()
                sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and wait_for_next == 0:
            particles.append(next_particle.release(space, shape_to_particle))
            wait_for_next = NEXT_DELAY

    next_particle.set_x(pygame.mouse.get_pos()[0])

    if wait_for_next > 1:
        wait_for_next -= 1
    elif wait_for_next == 1:
        next_particle = PreParticle(next_particle.x, rng.integers(0, 5))
        wait_for_next -= 1

    # Draw background and particles
    screen.fill(BG_COLOR)
    if wait_for_next == 0:
        next_particle.draw(screen)
    for w in walls:
        w.draw(screen)
    for p in particles:
        p.draw(screen)
        if p.pos[1] < PAD[1] and p.has_collided:
            label = overfont.render("Game Over!", 1, (0, 0, 0))
            screen.blit(label, PAD)
            game_over = True
    label = scorefont.render(f"Score: {handler.data['score']}", 1, (0, 0, 0))
    screen.blit(label, (10, 10))

    space.step(1/FPS)
    pygame.display.update()
    clock.tick(FPS)
