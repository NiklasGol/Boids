"""
Run simulation with visual output.
"""

import numpy as np
import random
from random import choice
import pyglet
from pyglet.gl import (
    Config, glScalef,
    glEnable, glBlendFunc, glLoadIdentity, glClearColor,
    GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_COLOR_BUFFER_BIT)
from pyglet.window import key

from .boid import Boid
from .predator import Predator
from .attractor import Attractor
from .obstacle import Obstacle
from .vector import distance

def create_random_obstacle(width, height, mask, max_size):
    position = np.array([random.uniform(0, width), random.uniform(0, height)])
    size=random.uniform(0, max_size)
    obj = Obstacle(position=position, size=size)
    mask = cut_from_mask(mask, position, size)
    return obj, mask

def cut_from_mask(mask, position, size):
    """ Mask is set over the hole are to avoid boids initialized in object"""
    width = mask.shape[0]
    height = mask.shape[1]
    position = np.asarray(position).astype(int)
    size = np.round(size+1).astype(int)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    dist = np.linalg.norm(np.concatenate(grid - position),axis=1)
    dist = dist.reshape([width, height])
    mask[dist<size] = 0
    return mask

def create_random_boid(mask):
    width = mask.shape[0]
    height = mask.shape[1]
    mask = mask.astype(bool)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    gridpoints = grid[mask]
    select_grid_point = int(random.random()*gridpoints.shape[0])
    position = gridpoints[select_grid_point] + np.array([random.random(),\
                            random.random()])
    return Boid(
        position=[random.uniform(0, width), random.uniform(0, height)],
        bounds=[width, height],
        velocity=np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0,\
                            50.0)]),
        age=random.uniform(0,1))

def create_centered_boid_flock(mask):
    width = mask.shape[0]
    height = mask.shape[1]
    mask = mask.astype(bool)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    gridpoints = grid[mask]
    select_grid_point = int(random.random()*gridpoints.shape[0])
    position = gridpoints[select_grid_point] + np.array([random.random(),\
                            random.random()])
    return Boid(
        position=[random.uniform(3*width/8, 5*width/8),\
                    random.uniform(3*height/8, 5*height/8)],
        bounds=[width, height],
        velocity=np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0,\
                            50.0)]),
        age=random.uniform(0,1))

def create_random_predator(mask):
    width = mask.shape[0]
    height = mask.shape[1]
    mask = mask.astype(bool)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    gridpoints = grid[mask]
    select_grid_point = int(random.random()*gridpoints.shape[0])
    position = gridpoints[select_grid_point] + np.array([random.random(),\
                            random.random()])
    return Predator(
        position=[random.uniform(0, width), random.uniform(0, height)],
        bounds=[width, height],
        velocity=np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0,\
                            50.0)]))

def create_predator_circle(mask):
    width = mask.shape[0]
    height = mask.shape[1]
    mask = mask.astype(bool)
    grid = np.moveaxis(np.mgrid[:width,:height], 0, -1)
    gridpoints = grid[mask]
    select_grid_point = int(random.random()*gridpoints.shape[0])
    position = gridpoints[select_grid_point] + np.array([random.random(),\
                            random.random()])
    choice_x = choice([(width/8, 2*width/8), (6*width/8, 7*width/8)])
    choice_y = choice([(height/8, 2*height/8), (6*height/8, 7*height/8)])
    return Predator(
        position=[random.uniform(*choice_x), random.uniform(*choice_y)],
        bounds=[width, height],
        velocity=np.array([random.uniform(-50.0, 50.0), random.uniform(-50.0,\
                            50.0)]))

def delete_catched_boid(boids, predators):
    removed_boids = []
    for boid in boids:
        for predator in predators:
            dist = distance(boid.position, predator.position)
            if dist <= 2*boid.size:
                removed_boids.append(boid)
    return removed_boids

def get_window_config():
    platform = pyglet.window.get_platform()
    display = platform.get_default_display()
    screen = display.get_default_screen()

    template = Config(double_buffer=True, sample_buffers=1, samples=4)
    try:
        config = screen.get_best_config(template)
    except pyglet.window.NoSuchConfigException:
        template = Config()
        config = screen.get_best_config(template)

    return config

def update_without_visual(dt, predators, boids, obstacles):
    attractors = []
    deleted_boids_timestep = []
    boids_to_delete = delete_catched_boid(boids, predators)
    # deleted_boids.append(boids_to_delete)
    for boid in boids_to_delete:
        if boid in boids:
            deleted_boids_timestep.append(boid.age)
            boids.remove(boid)
            print(boid)
    for boid in boids:
        boid.update(dt, boids, obstacles, predators)
    for predator in predators:
        predator.update(dt, attractors, predators, boids, obstacles)
    return deleted_boids_timestep, predators, boids

def run(config):
    deleted_boids = []

    glScalef(0.5, 0.5, 0.5)
    show_debug = False
    show_vectors = False

    n_boids = config["n_boids"]
    boids = []

    n_predators = config["n_predators"]
    predators = []

    attractors = []

    n_obstacles = config["n_obstacles"]
    max_size = config["max_size"]
    obstacles = []

    mouse_location = (0, 0)
    window = pyglet.window.Window(
        fullscreen=True,
        caption="Boids Simulation",
        config=get_window_config())

    #initialize
    mask = np.ones([window.width, window.height])

    for i in range(1, n_obstacles):
        obj, mask = create_random_obstacle(window.width, window.height, mask,\
                                            max_size)
        obstacles.append(obj)

    for i in range(n_boids):
        boids.append(create_centered_boid_flock(mask))

    for i in range(n_predators):
        predators.append(create_predator_circle(mask))

    def update(dt):
        dt = 0.1
        boids_to_delete = delete_catched_boid(boids, predators)
        for boid in boids_to_delete:
            if boid in boids:
                deleted_boids.append(boid.age)
                boids.remove(boid)
        for boid in boids:
            boid.update(dt, boids, attractors, obstacles, predators)
        for predator in predators:
            predator.update(dt, predators, boids, obstacles)
        # if len(deleted_boids) > 9:
        #     pyglet.app.exit()


        # schedule world updates as often as possible
    pyglet.clock.schedule(update)

    @window.event
    def on_draw():
        glClearColor(0.1, 0.1, 0.1, 1.0)
        window.clear()
        glLoadIdentity()

        for boid in boids:
            boid.draw(show_velocity=show_debug, show_view=show_debug,\
                        show_vectors=show_vectors)

        for predator in predators:
            predator.draw(show_velocity=show_debug, show_view=show_debug,\
                            show_vectors=show_vectors)

        for attractor in attractors:
            attractor.draw()

        for obstacle in obstacles:
            obstacle.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.Q:
            pyglet.app.exit()
        elif symbol == key.MINUS and len(boids) > 0:
            boids.pop()
        elif symbol == key.D:
            nonlocal show_debug
            show_debug = not show_debug
        elif symbol == key.V:
            nonlocal show_vectors
            show_vectors = not show_vectors
        elif symbol == key.A:
            attractors.append(Attractor(position=mouse_location))
        elif symbol == key.O:
            obstacles.append(Obstacle(position=mouse_location))

    @window.event
    def on_mouse_drag(x, y, *args):
        nonlocal mouse_location
        mouse_location = x, y

    @window.event
    def on_mouse_motion(x, y, *args):
        nonlocal mouse_location
        mouse_location = x, y

    pyglet.app.run()
