"""
Run simulation without visual output.
"""

import numpy as np
import random
from random import choice

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

def create_centered_boid_flock(mask):
    width = mask.shape[0]
    height = mask.shape[1]
    #print("# DEBUG: bounds: ", [width, height])
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

def update(dt, boids, predators, obstacles):
    attractors = []
    deleted_boids = []
    boids_to_delete = delete_catched_boid(boids, predators)
    for boid in boids_to_delete:
        if boid in boids:
            deleted_boids.append(boid.age)
            # print("len(tmp_deleted_boids): ", len(deleted_boids))
            boids.remove(boid)
    for boid in boids:
        boid.update(dt, boids, attractors, obstacles, predators)
    for predator in predators:
        predator.update(dt, predators, boids, obstacles)
    return boids, predators, deleted_boids


def run_bg(config, index):
    boids = []
    deleted_boids = []
    predators = []
    obstacles = []

    # initialize
    mask = np.ones(config["domain"])

    for i in range(config["n_obstacles"]):
        obj, mask = create_random_obstacle(config["domain"][0],\
                                        config["domain"][1], mask,\
                                        config["max_size"])
        obstacles.append(obj)

    for i in range(config["n_boids"]):
        boids.append(create_centered_boid_flock(mask))

    for i in range(config["n_predators"]):
        predators.append(create_predator_circle(mask))

    # run
    running = True
    deleted_boids = []
    while running:
        boids, predators, tmp_deleted_boids = update(config["dt"], boids,\
                                                        predators, obstacles)
        deleted_boids.extend(tmp_deleted_boids)
        if len(deleted_boids) > config["deleted_boids_max"]:
            print("Run is done, index: ", index)
            running = False
    return deleted_boids[:config["deleted_boids_max"]]
