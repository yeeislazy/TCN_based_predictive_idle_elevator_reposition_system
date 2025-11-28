"""
Optional pygame GUI for elevator simulator.
This file is intentionally small: if pygame isn't installed, the simulator still runs.
To enable GUI, you can import and call `start_gui(loop_callback)` where loop_callback is a function returning current state.

For this initial scaffold the GUI is a placeholder that can be extended.
"""
try:
    import pygame
except Exception:
    pygame = None


def start_gui(get_state_callback, width=400, height=600):
    if pygame is None:
        print('pygame not installed; GUI unavailable')
        return
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        state = get_state_callback()
        screen.fill((30, 30, 30))
        # very simple drawing: floors and elevator
        floors = state.get('floors', 10)
        current_floor = state.get('current_floor', 1)
        fh = height / floors
        for f in range(1, floors+1):
            y = height - f*fh
            pygame.draw.rect(screen, (50, 50, 50), (50, y, 300, fh-2))
            if f == current_floor:
                pygame.draw.rect(screen, (200, 80, 80), (180, y+5, 40, fh-12))
        pygame.display.flip()
        clock.tick(10)
    pygame.quit()

if __name__ == '__main__':
    # Example usage with dummy state
    import time
    def dummy_state():
        t = int(time.time()) % 10 + 1
        return {'floors': 10, 'current_floor': t}
    start_gui(dummy_state)
