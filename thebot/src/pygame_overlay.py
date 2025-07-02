import pygame
import sys
import threading
import time
import numpy as np

class PygameOverlay:
    def __init__(self, width=800, height=600, fov=None):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI Aim Assist Overlay")
        self.clock = pygame.time.Clock()
        self.running = False
        self.fov = fov or 280
        self.status = "Idle"
        self.detections = []
        self.fps = 0
        self.crosshair_color = (0, 255, 0)
        self.font = pygame.font.SysFont("consolas", 18)
        self.menu_active = False
        self.menu_items = ["Toggle Aim Assist (F1)", "Exit (ESC)"]
        self.selected_menu = 0

    def set_detections(self, detections):
        self.detections = detections

    def set_status(self, status):
        self.status = status

    def set_fps(self, fps):
        self.fps = fps

    def draw_overlay(self):
        self.screen.fill((30, 30, 30))
        # Draw FOV region
        fov_rect = pygame.Rect((self.width-self.fov)//2, (self.height-self.fov)//2, self.fov, self.fov)
        pygame.draw.rect(self.screen, (0, 128, 255), fov_rect, 2)
        # Draw detections (bounding boxes)
        for det in self.detections:
            x1, y1, x2, y2 = map(int, det.get('box', (0,0,0,0)))
            conf = det.get('confidence', 0)
            class_id = det.get('class_id', -1)
            pygame.draw.rect(self.screen, (255,0,0), (x1, y1, x2-x1, y2-y1), 2)
            label = f"ID:{class_id} {conf:.2f}"
            text = self.font.render(label, True, (255,255,255))
            self.screen.blit(text, (x1, y1-18))
        # Draw crosshair
        cx, cy = self.width//2, self.height//2
        pygame.draw.line(self.screen, self.crosshair_color, (cx-10, cy), (cx+10, cy), 2)
        pygame.draw.line(self.screen, self.crosshair_color, (cx, cy-10), (cx, cy+10), 2)
        # Draw status and FPS
        status_text = self.font.render(f"Status: {self.status}", True, (255,255,0))
        fps_text = self.font.render(f"FPS: {self.fps:.1f}", True, (0,255,0))
        self.screen.blit(status_text, (10, 10))
        self.screen.blit(fps_text, (10, 35))
        # Draw menu if active
        if self.menu_active:
            menu_bg = pygame.Surface((300, 100))
            menu_bg.set_alpha(200)
            menu_bg.fill((50,50,50))
            self.screen.blit(menu_bg, (self.width//2-150, self.height//2-50))
            for i, item in enumerate(self.menu_items):
                color = (255,255,255) if i != self.selected_menu else (0,255,255)
                item_text = self.font.render(item, True, color)
                self.screen.blit(item_text, (self.width//2-130, self.height//2-40 + i*30))
        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_F1:
                    self.status = "Aim Assist: Toggled"
                elif event.key == pygame.K_TAB:
                    self.menu_active = not self.menu_active
                elif self.menu_active:
                    if event.key == pygame.K_UP:
                        self.selected_menu = (self.selected_menu - 1) % len(self.menu_items)
                    elif event.key == pygame.K_DOWN:
                        self.selected_menu = (self.selected_menu + 1) % len(self.menu_items)
                    elif event.key == pygame.K_RETURN:
                        if self.selected_menu == 0:
                            self.status = "Aim Assist: Toggled"
                        elif self.selected_menu == 1:
                            self.running = False

    def run(self):
        self.running = True
        while self.running:
            self.handle_events()
            self.draw_overlay()
            self.fps = self.clock.get_fps()
            self.clock.tick(60)
        pygame.quit()
        sys.exit()

    def cleanup(self):
        # Clean up pygame and set running to False
        self.running = False
        try:
            pygame.quit()
        except Exception:
            pass
        # Optionally, add more cleanup logic if needed

# Example usage:
if __name__ == "__main__":
    overlay = PygameOverlay(width=960, height=720, fov=280)
    # Simulate detections for demo
    overlay.set_detections([
        {'box': (400, 300, 500, 400), 'confidence': 0.92, 'class_id': 0},
        {'box': (600, 200, 700, 350), 'confidence': 0.81, 'class_id': 0}
    ])
    overlay.set_status("Connected to Arduino")
    overlay.run()
