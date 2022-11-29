import numpy as np
import pywavefront
import OpenGL.GL as gl


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