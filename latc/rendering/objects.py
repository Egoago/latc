import ctypes

import numpy as np
import pywavefront
import OpenGL.GL as gl

from latc import utils, Renderer


def scale(T, s):
    if isinstance(s, list):
        s = np.array(s)
    s = utils.homogeneous(s)
    return T @ np.diag(s)


def translate(T, t):
    _T = np.identity(4, dtype=float)
    _T[:3, -1] = t
    return T @ _T


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

    def render(self, renderer: Renderer, M=None):
        raise NotImplementedError

    def model_mtx(self, M=None, renderer: Renderer = None) -> np.ndarray:
        if M is None:
            M = np.identity(4, dtype=float)
        M = translate(M, self.translation)
        M = scale(M, self.scale)
        if renderer is not None:
            renderer.set_uniform('M', M)
        return M


class Object(Node):
    def __init__(self, meshes, scale, translation):
        super().__init__(scale=scale, translation=translation)
        self.meshes = meshes

    def render(self, renderer: Renderer, M=None):
        M = self.model_mtx(M)
        for mesh in self.meshes:
            mesh.render(renderer, M)


class TriangleMesh(Node):

    @staticmethod
    def stack_data(vertices, faces, normals):
        data = []
        for face_i, face in enumerate(faces):
            for vertex_i in face:
                data.append(np.concatenate([vertices[vertex_i],  normals[face_i]]))
        return np.array(data, dtype=np.float32)

    def __init__(self, faces, vertices, scale, translation, normals, wireframe=False):
        super().__init__(scale=scale, translation=translation)
        self.triangle_count = len(faces)
        self.wireframe = wireframe
        self.vao = gl.glGenVertexArrays(1)
        vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glEnableVertexAttribArray(0)
        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        float_size = 8
        offset = 3*float_size
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, offset, None)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, offset, ctypes.cast(offset//2, ctypes.c_void_p))
        data = self.stack_data(vertices, faces, normals)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data, gl.GL_STATIC_DRAW)

    def render(self, renderer: Renderer, M=None):
        renderer.set_uniform('wireframe', self.wireframe)
        if self.wireframe:
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
            gl.glDisable(gl.GL_CULL_FACE)
        gl.glBindVertexArray(self.vao)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if self.wireframe else gl.GL_FILL)
        self.model_mtx(M, renderer)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.triangle_count*3)
        gl.glBindVertexArray(0)
        if self.wireframe:
            gl.glEnable(gl.GL_CULL_FACE)


class Box(TriangleMesh):
    vertices = np.array([[-1, -1, 1], [1, -1, 1],
                         [1, 1, 1], [-1, 1, 1],
                         [-1, -1, -1], [1, -1, -1],
                         [1, 1, -1], [-1, 1, -1]], dtype=np.float32) / 2.
    normals = np.array([[0, 0, 1],
                        [0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 0, -1],
                        [0, 0, -1],
                        [-1, 0, 0],
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, -1, 0],
                        [0, 1, 0],
                        [0, 1, 0]], dtype=np.float32)
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

    def __init__(self, scale, translation, wireframe=False):
        super().__init__(faces=Box.faces, vertices=Box.vertices,
                         scale=scale, translation=translation,
                         wireframe=wireframe, normals=Box.normals)


class LoadedObject(Object):

    @staticmethod
    def clamp_to_bbox(vertex, bbox):
        center = [(bbox[1][i] + bbox[0][i]) / 2. for i in range(3)]
        ranges = [abs(bbox[1][i] - bbox[0][i]) for i in range(3)]
        scale = max(ranges)
        return [(vertex[i] - center[i]) / scale for i in range(3)]
    @staticmethod
    def generate_normals(vertices, faces):
        normals = []
        for face in faces:
            vs = np.array([vertices[i] for i in face])
            a, b, c = vs[0, :], vs[1, :], vs[2, :]
            normals.append(np.cross(b-a, c-a))
        return utils.normalize(np.array(normals, dtype=np.float32))

    def __init__(self, file_path, scale=None, translation=None):
        obj = pywavefront.Wavefront(file_path, collect_faces=True, create_materials=True, )
        self.vertices = obj.vertices
        bbox = (self.vertices[0], self.vertices[0])
        for vertex in self.vertices:
            min_v = [min(bbox[0][i], vertex[i]) for i in range(3)]
            max_v = [max(bbox[1][i], vertex[i]) for i in range(3)]
            bbox = (min_v, max_v)
        self.vertices = [LoadedObject.clamp_to_bbox(vertex, bbox) for vertex in self.vertices]

        meshes = []
        for mesh in obj.mesh_list:
            normals = self.generate_normals(self.vertices, mesh.faces)
            meshes.append(TriangleMesh(vertices=self.vertices, faces=mesh.faces,
                                       scale=None, translation=None, normals=normals))
        super().__init__(meshes=meshes, scale=scale, translation=translation)
