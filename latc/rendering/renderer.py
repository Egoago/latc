import math

import numpy as np
import pygame
import OpenGL.GL as gl
from OpenGL.GL import shaders
import OpenGL.GLU as glu
import pywavefront

from latc import utils


class Node:
    def __init__(self, scale=None, translation=None):
        if scale is None:
            scale = [1., 1., 1.]
        if isinstance(scale, float):
            scale = [scale, scale, scale]

        if translation is None:
            translation = [0., 0., 0.]
        self.scale = scale
        self.translation = translation

    def render_impl(self):
        raise NotImplementedError()

    def apply(self):
        gl.glPushMatrix()
        gl.glTranslatef(*self.translation)
        gl.glScalef(*self.scale)
        self.render_impl()
        gl.glPopMatrix()


class Object(Node):
    def __init__(self, meshes, scale, translation):
        super().__init__(scale=scale, translation=translation)
        self.meshes = meshes

    def render_impl(self):
        for mesh in self.meshes:
            mesh.apply()


class TriangleMesh(Node):
    def __init__(self, faces, vertices, scale, translation):
        super().__init__(scale=scale, translation=translation)
        self.faces = faces
        self.vertices = vertices

    def render_impl(self):
        gl.glBegin(gl.GL_TRIANGLES)
        for face in self.faces:
            for vertex_i in face:
                gl.glVertex3fv(self.vertices[vertex_i])
        gl.glEnd()


class Box(TriangleMesh):
    vertices = np.array([[-1, -1, 1], [1, -1, 1],
                         [1, 1, 1], [-1, 1, 1],
                         [-1, -1, -1], [1, -1, -1],
                         [1, 1, -1], [-1, 1, -1]], dtype=np.float32) / 2.
    faces = np.array([[0, 1, 2],
                      [2, 3, 0],
                      [1, 5, 6],
                      [6, 2, 1],
                      [5, 4, 7],
                      [7, 6, 5],
                      [4, 0, 3],
                      [3, 7, 4],
                      [4, 5, 1],
                      [1, 0, 4],
                      [3, 2, 6],
                      [6, 7, 3]], dtype=np.uint32)

    def __init__(self, scale, translation):
        super().__init__(faces=Box.faces, vertices=Box.vertices,
                         scale=scale, translation=translation)


class Tunnel(Object):
    def __init__(self, size: utils.Size, scale=None, translation=None, count=5):
        depth = min(size.width, size.height)
        meshes = [Box(scale=[size.width, size.height, depth],
                      translation=[0., 0., -depth * (0.5 + 2 * i)]) for i in range(count)]
        super().__init__(meshes,
                         scale=scale, translation=translation)


class LoadedObject(Object):

    @staticmethod
    def clamp_to_bbox(vertex, bbox):
        center = [(bbox[1][i] + bbox[0][i]) / 2. for i in range(3)]
        ranges = [abs(bbox[1][i] - bbox[0][i]) for i in range(3)]
        scale = max(ranges)
        return [(vertex[i] - center[i]) / scale for i in range(3)]

    def __init__(self, file_path, scale=None, translation=None):
        obj = pywavefront.Wavefront(file_path, collect_faces=True, create_materials=True)
        self.vertices = obj.vertices
        bbox = (self.vertices[0], self.vertices[0])
        for vertex in self.vertices:
            min_v = [min(bbox[0][i], vertex[i]) for i in range(3)]
            max_v = [max(bbox[1][i], vertex[i]) for i in range(3)]
            bbox = (min_v, max_v)
        self.vertices = [LoadedObject.clamp_to_bbox(vertex, bbox) for vertex in self.vertices]

        meshes = []
        for mesh in obj.mesh_list:
            meshes.append(TriangleMesh(vertices=self.vertices, faces=mesh.faces,
                                       scale=None, translation=None))
        super().__init__(meshes=meshes, scale=scale, translation=translation)


class Renderer(Node):
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

    def __init__(self, objects, config: utils.Calibration, debug=False):
        self.objects = objects
        pygame.init()
        vertex_shader = '''#version 120
                        void main() {
                            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
                        }'''
        fragment_shader = '''#version 120
                          void main() {
                            gl_FragColor = vec4(0, 0, 1, 1);
                          }'''
        # try:
        #     self.program = shaders.compileProgram(
        #         shaders.compileShader(vertex_shader, gl.GL_VERTEX_SHADER),
        #         shaders.compileShader(fragment_shader, gl.GL_FRAGMENT_SHADER), validate=False)
        # except Exception as e:
        #     print(str(e).encode('utf-8').decode('unicode_escape'))
        #     raise SystemExit()
        # gl.glUseProgram(self.program)
        self.resolution = pygame.display.list_modes()[0]
        self.window = pygame.display.set_mode(self.resolution, pygame.DOUBLEBUF | pygame.OPENGL | pygame.FULLSCREEN, 8)
        self.fovy = math.degrees(math.atan2(config.screen_size.height, config.near))

        self.PV = Renderer.create_projection_matrix(config)
        # glu.gluPerspective(self.fovy, (config.screen_size.width / config.screen_size.height), config.near, config.far)
        # gl.glTranslatef(0, 0, config.near)
        self.shear_mtx = np.identity(4)
        if debug:
            self.objects.append(Tunnel(config.screen_size))
        super().__init__()

    def __handle_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                quit()

    def render_impl(self):
        for object in self.objects:
            object.apply()

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
        self.apply()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        pygame.display.flip()
