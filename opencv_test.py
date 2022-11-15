import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu
import pywavefront

def run():
    scene = pywavefront.Wavefront('data/deer.obj', collect_faces=True, create_materials=True)

    scene_box = (scene.vertices[0], scene.vertices[0])
    for vertex in scene.vertices:
        min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
        max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
        scene_box = (min_v, max_v)

    scene_size = [scene_box[1][i] - scene_box[0][i] for i in range(3)]
    max_scene_size = max(scene_size)
    scaled_size = 5
    scene_scale = [scaled_size / max_scene_size for i in range(3)]
    scene_trans = [-(scene_box[1][i] + scene_box[0][i]) / 2 for i in range(3)]

    def Model():
        gl.glPushMatrix()
        gl.glScalef(*scene_scale)
        gl.glTranslatef(*scene_trans)

        for mesh in scene.mesh_list:
            gl.glBegin(gl.GL_TRIANGLES)
            for face in mesh.faces:
                for vertex_i in face:
                    gl.glVertex3f(*scene.vertices[vertex_i])
            gl.glEnd()

        gl.glPopMatrix()

    def main():
        pygame.init()
        modes = pygame.display.list_modes()
        pygame.display.set_mode(modes[0], pygame.DOUBLEBUF | pygame.OPENGL | pygame.FULLSCREEN, 8)
        glu.gluPerspective(45, (modes[0][0] / modes[0][1]), 1, 500.0)
        gl.glTranslatef(0.0, 0.0, -10)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_LEFT:
                        gl.glTranslatef(-0.5, 0, 0)
                    if event.key == pygame.K_RIGHT:
                        gl.glTranslatef(0.5, 0, 0)
                    if event.key == pygame.K_UP:
                        gl.glTranslatef(0, 1, 0)
                    if event.key == pygame.K_DOWN:
                        gl.glTranslatef(0, -1, 0)

            gl.glRotatef(1, 5, 1, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            Model()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            pygame.display.flip()
            pygame.time.wait(10)

    main()


if __name__ == "__main__":
    run()
