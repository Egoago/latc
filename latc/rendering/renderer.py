from typing import Union

import glfw
import numpy as np
import OpenGL.GL as gl
from OpenGL.GL import shaders

from latc import utils


class Renderer:
    @staticmethod
    def load_shaders():
        VERTEX_SHADER = """#version 330
                        uniform mat4 PV;
                        uniform mat4 S;
                        uniform mat4 M;
                        layout (location = 0) in vec3 position;
                        layout (location = 1) in vec3 normal;
                        out vec3 Normal;
                        out vec3 pWorld;
                        void main() {
                            vec4 p_world = M * vec4(position, 1.0f);
                            gl_Position = PV * S * p_world;
                            Normal = mat3(transpose(inverse(M))) * normal;
                            pWorld = vec3(p_world);
                        }"""
        FRAGMENT_SHADER = """#version 330
                          uniform vec3 lightDir;
                          uniform vec3 lightColor;
                          uniform vec3 eyeWorld;
                          uniform bool wireframe;
                          in vec3 Normal;
                          in vec3 pWorld;
                          void main() {
                            vec4 result;
                            if (wireframe)
                                result = vec4(1.0, 0.8, 0.0, 0.3);
                            else {
                                vec3 norm = normalize(Normal);
                                float diff = max(dot(norm, lightDir), 0.0);
                                vec3 diffuse = vec3(0.0, 0.1, 0.1)+diff * lightColor;
                                
                                vec3 viewDir = normalize(eyeWorld - pWorld);
                                vec3 reflectDir = reflect(-lightDir, norm);
                                float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
                                vec3 specular = spec * lightColor; 
                                
                                result = vec4(diffuse+specular, 1.0);
                            }
                            gl_FragColor = vec4(result.xyz * exp2(0.03*pWorld.z), result.a);
                          }"""
        program = shaders.compileProgram(shaders.compileShader(VERTEX_SHADER, gl.GL_VERTEX_SHADER),
                                         shaders.compileShader(FRAGMENT_SHADER, gl.GL_FRAGMENT_SHADER))
        gl.glUseProgram(program)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glLineWidth(4)
        return program

    @staticmethod
    def create_PV_mtx(config, window_rect=None) -> np.ndarray:
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

    def init_window(self):
        if not glfw.init():
            raise RuntimeError("GLFW initialization error")
        glfw.window_hint(glfw.SAMPLES, 4)
        window = glfw.create_window(self.config.screen_res.width,
                                    self.config.screen_res.height,
                                    "Latc", glfw.get_primary_monitor(), None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not open window")
        glfw.make_context_current(window)
        glfw.set_key_callback(window, self.key_callback)
        return window

    def __init__(self, objects, config: utils.Calibration):
        self.objects = objects
        self.config = config
        self.window = self.init_window()
        self.program = self.load_shaders()

        self.PV = Renderer.create_PV_mtx(config)
        self.set_uniform('PV', self.PV)
        self.update_shear(np.array([0., 0., config.screen_size.height]))
        self.light_dir = utils.normalize(np.array([1., 1., 0.5]))
        self.light_color = np.array([1., 0.7, 0.4])

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE:
            glfw.terminate()
            exit()

    def render(self):
        glfw.poll_events()
        self.set_uniform('lightDir', self.light_dir)
        self.set_uniform('lightColor', self.light_color)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        for object in self.objects:
            object.render(self)
        glfw.swap_buffers(self.window)

    def set_uniform(self, name, data: Union[np.ndarray, bool]):
        uniform = gl.glGetUniformLocation(self.program, name)
        if isinstance(data, bool):
            gl.glUniform1i(uniform, data)
        elif data.ndim == 2:
            gl.glUniformMatrix4fv(uniform, 1, True, data)
        elif data.shape[0] == 3:
            gl.glUniform3fv(uniform, 1, data)
        else:
            gl.glUniform4fv(uniform, 1, data)


    def update_shear(self, eye_world):
        eye_depth = eye_world[2]
        eye_vcam = eye_world - np.array([0, 0, self.config.screen_size.height / 2.], float)  # TODO y?
        shear_mtx = np.array([[1., 0., -eye_vcam[0] / eye_depth, 0.],
                              [0., 1., -eye_vcam[1] / eye_depth, 0.],
                              [0., 0., 1 - eye_vcam[2] / eye_depth, 0.],
                              [0., 0., 0., 1.]], float)
        self.set_uniform('S', shear_mtx)
        self.set_uniform('eyeWorld', eye_world)
