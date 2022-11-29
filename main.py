from datetime import datetime

import numpy as np

import latc

def create_tunnel(size):
    depth = min(size.width, size.height)
    return [latc.objects.Box(scale=[size.width, size.height, depth],
                             translation=[0., 0., -depth * (0.5 + 2 * i)]) for i in range(4)]
if __name__ == "__main__":
    config = latc.Calibration.load_yaml('data/config.yaml')
    tracker = latc.MediapipeTracker(config, latc.CVWebCam())
    objects = [latc.objects.LoadedObject('data/deer.obj', config.screen_size.height, [0, 0, -1.5 * config.screen_size.height])]
    objects += create_tunnel(config.screen_size)
    renderer = latc.Renderer(objects, config)
    while True:
        eye_world = tracker.update()
        print(eye_world)
        # seconds = datetime.now().second
        # eye_world = np.array([config.screen_size.width / 2,
        #                       config.screen_size.height / 2,
        #                       config.screen_size.height], dtype=float)
        # if seconds//10 % 3 == 1:
        #     eye_world[1] *= -1
        # elif seconds // 10 % 3 == 2:
        #     eye_world[0] *= -1
        if eye_world is not None:
            renderer.update_shear(eye_world=eye_world)

        renderer.render()
