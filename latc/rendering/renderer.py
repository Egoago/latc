import numpy as np
import pygame
import OpenGL.GL as gl
from OpenGL.GL import shaders

from latc import utils


class Renderer:
    @staticmethod
    def compile_program(vertex_source, fragment_source):
        VERTEX_SHADER = shaders.compileShader(vertex_source, gl.GL_VERTEX_SHADER)
        FRAGMENT_SHADER = shaders.compileShader(fragment_source, gl.GL_FRAGMENT_SHADER)
        return shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

    @staticmethod
    def create_projection_matrix(config, window_rect=None) -> np.ndarray:
        # TODO window_rect
        w, h = config.screen_size.width, config.screen_size.height
        r, t = w / 2., h / 2.
        l, b = -r, -t
        aspect = config.screen_size.width / config.screen_size.height
        n, f = config.near, config.far
        P = np.array([[n / aspect, 0, (r + l) / w, 0],
                      [0, n, (t + b) / h, 0],
                      [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                      [0, 0, -1, 0]], float)
        V = np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., -t],
                      [0., 0., 0., 1.]], float)
        return P @ V

    def __init__(self, objects, config: utils.Calibration):
        self.objects = objects
        self.config = config
        pygame.init()

        self.resolution = pygame.display.list_modes()[0]
        self.window = pygame.display.set_mode(self.resolution, pygame.DOUBLEBUF | pygame.OPENGL | pygame.FULLSCREEN, 8)

        self.PV = Renderer.create_projection_matrix(config)
        self.shear_mtx = np.identity(4)
        super().__init__()

    def __handle_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                quit()

    def __load_PVS(self):
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadMatrixf(self.PV.T.flatten())
        gl.glMultMatrixf(self.shear_mtx.T.flatten())
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def render(self):
        self.__handle_keys()
        self.__load_PVS()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        for object in self.objects:
            object.apply()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        pygame.display.flip()

    def update_shear(self, eye_world):
        eye_depth = eye_world[2]
        eye_vcam = eye_world - np.array([0, 0, self.config.screen_size.height / 2.], float)  # TODO y?
        self.shear_mtx = np.array([[1., 0., -eye_vcam[0] / eye_depth, 0.],
                                   [0., 1., -eye_vcam[1] / eye_depth, 0.],
                                   [0., 0., 1 - eye_vcam[2] / eye_depth, 0.],
                                   [0., 0., 0., 1.]], float)
