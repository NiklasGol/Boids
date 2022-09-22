# -*- coding: utf-8 -*-

import math
import numpy as np
from pyglet.gl import (
    glPushMatrix, glPopMatrix, glBegin, glEnd, glColor3f,
    glVertex2f, glTranslatef, glRotatef,
    GL_LINE_LOOP, GL_LINES, GL_TRIANGLES)

from . import vector

_PREDATOR_RANGE = 1000.0
_PREDATOR_VIEW_ANGLE = 110
_PREDATOR_COLLISION_DISTANCE = 45.0
_OBSTACLE_COLLISION_DISTANCE = 250.0
_MAX_COLLISION_VELOCITY = 1.0
_CHANGE_VECTOR_LENGTH = 15.0
_MAX_SPEED = 158.0
_MIN_SPEED = 55.0
_BOUNDARY_SLOP = 0.0

_PREDATOR_AVOIDANCE_FACTOR = 30
_OBSTACLE_AVOIDANCE_FACTOR = 200.0
_ATTACK_FACTOR = 100.00

class Predator:
    def __init__(self,
                 position=[100.0, 100.0],
                 bounds=[1000, 1000],
                 velocity=np.array([0.0, 0.0]),
                 size=10.0,
                 color=[1.0, 0.0, 0.0]):
        self.position = position
        self.bounds = bounds
        self.wrap_bounds = [i + _BOUNDARY_SLOP for i in bounds]
        self.velocity = velocity
        self.size = size
        self.color = color
        self.change_vectors = []

    def __repr__(self):
        return "Predator: position={}, velocity={}, color={}".format(\
            self.position, self.velocity, self.color)


    def render_velocity(self):
        glColor3f(0.6, 0.6, 0.6)
        glBegin(GL_LINES)
        glVertex2f(0.0, 0.0)
        glVertex2f(0.0, _PREDATOR_RANGE)
        glEnd()


    def render_view(self):
        glColor3f(0.6, 0.1, 0.1)
        glBegin(GL_LINE_LOOP)

        step = 10
        # render a circle for the boid's view
        for i in range(-_PREDATOR_VIEW_ANGLE, _PREDATOR_VIEW_ANGLE + step, step):
            glVertex2f(_PREDATOR_RANGE * math.sin(math.radians(i)),
                       (_PREDATOR_RANGE * math.cos(math.radians(i))))
        glVertex2f(0.0, 0.0)
        glEnd()


    def render_change_vectors(self):
        glBegin(GL_LINES)

        color = [0.0, 0.0, 0.0]
        for i, (factor, vec) in enumerate(self.change_vectors):
            color[i % 3] = 1.0
            glColor3f(*color)
            glVertex2f(0.0, 0.0)
            glVertex2f(*[i * factor * _CHANGE_VECTOR_LENGTH for i in vec])
            color[i % 3] = 0.0
        glEnd()


    def render_boid(self):
        glBegin(GL_TRIANGLES)
        glColor3f(*self.color)
        glVertex2f(-(self.size), 0.0)
        glVertex2f(self.size, 0.0)
        glVertex2f(0.0, self.size * 3.0)
        glEnd()


    def draw(self, show_velocity=False, show_view=False, show_vectors=False):
        glPushMatrix()
        # apply the transformation for the boid
        glTranslatef(self.position[0], self.position[1], 0.0)
        if show_vectors:
            self.render_change_vectors()

        glRotatef(math.degrees(math.atan2(self.velocity[0], self.velocity[1])),\
                0.0, 0.0, -1.0)

        # render the boid's velocity
        if show_velocity:
            self.render_velocity()

        # render the boid's view
        if show_view:
            self.render_view()

        # render the boid itself
        self.render_boid()
        glPopMatrix()


    # hier muss was geändert werden die IF bedingung muss wieder rein damit die
    # Predators die Boids erst ab eine gewissen Distanz sehen
    # vielleicht die Colision avoidance doch über Winkelzu Obkjekt machen
    def determine_nearest_boid(self, all_boids):
        """Note, this can be done more efficiently if performed globally,
        rather than for each individual boid.
        Add viewing angle later.
        """
        boids_in_view = []
        boids_in_view_distance = []
        for boid in all_boids:
            relevant_position = vector.relevant_position(self.position,\
                                                boid.position, self.bounds)
            diff = (relevant_position[0] - self.position[0],\
                    relevant_position[1] - self.position[1])
            if (vector.magnitude(*diff) <= _PREDATOR_RANGE and\
                vector.angle_between(self.velocity, diff) <=\
                _PREDATOR_VIEW_ANGLE):
                boids_in_view_distance.append(vector.magnitude(*diff))
                boids_in_view.append(boid)
        if not boids_in_view:
            return []
        else:
            arg_min = np.argmin(np.asarray(boids_in_view_distance))
            return boids_in_view[arg_min]

    def determine_nearby_predators(self, all_predators):
        """Note, this can be done more efficiently if performed globally,
        rather than for each individual boid.
        """

        for pred in all_predators:
            relevant_position = vector.relevant_position(self.position,\
                                                pred.position, self.bounds)
            diff = (relevant_position[0] - self.position[0],\
                    relevant_position[1] - self.position[1])
            if (pred != self and
                    vector.magnitude(*diff) <= _PREDATOR_RANGE and\
                    vector.angle_between(self.velocity, diff) <=\
                    _PREDATOR_VIEW_ANGLE):
                yield pred
        return


    def determine_nearby_objects(self, all_objects, collision_distance):
        for obj in all_objects:
            relevant_position = vector.relevant_position(self.position,\
                                                    obj.position, self.bounds)
            diff = (relevant_position[0] - self.position[0],\
                    relevant_position[1] - self.position[1])
            if (obj != self and
                    vector.magnitude(*diff) <= collision_distance):
                yield obj
        return

    def avoid_collisions(self, objs, collision_distance):
        # determine nearby predetors using distance only
        nearby_objs = self.determine_nearby_objects(objs, collision_distance)

        c = [0.0, 0.0]
        for obj in nearby_objs:
            relevant_position = vector.relevant_position(self.position,\
                                                obj.position, self.bounds)
            diff = (relevant_position[0] - self.position[0],\
                    relevant_position[1] - self.position[1])
            inv_sqr_magnitude = 1/vector.magnitude(*diff)

            c[0] = c[0] - inv_sqr_magnitude * diff[0]
            c[1] = c[1] - inv_sqr_magnitude * diff[1]
        # return vector.limit_magnitude(c, _MAX_COLLISION_VELOCITY)
        return c


    def attack(self, nearest_boid):
        # generate a vector that moves the boid towards the attractors
        # print("# DEBUG: nearest_boid: ", nearest_boid)
        a = [0.0, 0.0]
        if not nearest_boid:
            return a
        else:
            relevant_position = vector.relevant_position(self.position,\
                                            nearest_boid.position, self.bounds)
            diff = [relevant_position[0] - self.position[0],\
                    relevant_position[1] - self.position[1]]
            magnitude = vector.magnitude(*diff)
            a[0] += diff[0]/magnitude
            a[1] += diff[1]/magnitude
            return a


    def update(self, dt, all_predators, all_boids, obstacles):
        nearest_boid = self.determine_nearest_boid(all_boids)
        nearby_predators = list(self.determine_nearby_predators(all_predators))

        # update the boid's direction based on several behavioural rules
        predator_avoidance_vector = self.avoid_collisions(all_predators,\
                                                _PREDATOR_COLLISION_DISTANCE)
        obstacle_avoidance_vector = self.avoid_collisions(obstacles,\
                                                _OBSTACLE_COLLISION_DISTANCE)
        attack_vector = self.attack(nearest_boid)

        self.change_vectors = [
            (_ATTACK_FACTOR, attack_vector),
            (_PREDATOR_AVOIDANCE_FACTOR, predator_avoidance_vector),
            (_OBSTACLE_AVOIDANCE_FACTOR, obstacle_avoidance_vector)]

        for factor, vec in self.change_vectors:
            self.velocity[0] += factor *vec[0]
            self.velocity[1] += factor *vec[1]

        # ensure that the boid's velocity is <= _MAX_SPEED
        self.velocity = vector.limit_magnitude(self.velocity, _MAX_SPEED,
                                                _MIN_SPEED)

        # move the boid to its new position, given its current velocity,
        # taking into account the world boundaries
        for i in range(0, len(self.position)):
            self.position[i] += dt * self.velocity[i]
            if self.position[i] >= self.wrap_bounds[i]:
                self.position[i] = (self.position[i] % self.wrap_bounds[i]) - _BOUNDARY_SLOP
            elif self.position[i] < -_BOUNDARY_SLOP:
                self.position[i] = self.position[i] + self.wrap_bounds[i] + _BOUNDARY_SLOP
