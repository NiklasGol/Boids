# -*- coding: utf-8 -*-

import math
import numpy as np
from pyglet.gl import (
    glPushMatrix, glPopMatrix, glBegin, glEnd, glColor3f,
    glVertex2f, glTranslatef, glRotatef,
    GL_LINE_LOOP, GL_LINES, GL_TRIANGLES)

from . import vector

_BOID_RANGE = 950.0
_BOID_VIEW_ANGLE = 110
_BOID_COLLISION_DISTANCE = 45.0
_OBSTACLE_COLLISION_DISTANCE = 250.0
_PREDATOR_COLLISION_DISTANCE = 300.0
_MAX_COLLISION_VELOCITY = 1.0
_CHANGE_VECTOR_LENGTH = 15.0
_MAX_SPEED = 150.0
_MIN_SPEED = 25.0
_BOUNDARY_SLOP = 0.0

# _COHESION_FACTOR = 0.03
_COHESION_FACTOR = 0.5
_ALIGNMENT_FACTOR = 0.045
# _BOID_AVOIDANCE_FACTOR = 7.5
_BOID_AVOIDANCE_FACTOR = 10
_OBSTACLE_AVOIDANCE_FACTOR = 200.0
_PREDATOR_COLLISION_FACTOR = 500.0
_ATTRACTOR_FACTOR = 0.0035


class Boid:
    def __init__(self,
                 position=[100.0, 100.0],
                 bounds=[1000, 1000],
                 velocity=np.array([0.0, 0.0]),
                 size=10.0,
                 color=[0.0, 0.0, 1.0],
                 age=0):
        self.position = position
        self.bounds = bounds
        self.velocity = velocity
        self.size = size
        self.color = color
        self.wrap_bounds = [i for i in bounds]
        self.change_vectors = []
        self.age = age
        self.viewing_range, self.viewing_angle, self.max_speed =\
            self.characteristics_based_on_age(self.age)

    def characteristics_based_on_age(self, age):
        viewing_range = _BOID_RANGE-20*(age-0.5)**2
        viewing_angle = _BOID_VIEW_ANGLE-20*(age-0.5)**2
        max_speed = _MAX_SPEED-200*(age-0.5)**2
        return viewing_range, viewing_angle, max_speed

    def __repr__(self):
        return "Boid: position={}, velocity={}, age={}".format(self.position,\
            self.velocity, self.age)


    def render_velocity(self):
        glColor3f(0.6, 0.6, 0.6)
        glBegin(GL_LINES)
        glVertex2f(0.0, 0.0)
        glVertex2f(0.0, self.viewing_range)
        glEnd()


    def render_view(self):
        glColor3f(0.6, 0.1, 0.1)
        glBegin(GL_LINE_LOOP)

        step = 10
        # render a circle for the boid's view
        for i in range(-self.viewing_angle, self.viewing_angle + step, step):
            glVertex2f(self.viewing_range * math.sin(math.radians(i)),
                       (self.viewing_range * math.cos(math.radians(i))))
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


    def determine_nearby_boids(self, all_boids):
        """
        Note, this can be done more efficiently if performed globally,
        rather than for each individual boid.
        """
        for boid in all_boids:
            relevant_position = vector.relevant_position(self.position,\
                                                    boid.position, self.bounds)
            diff = (relevant_position[0] - self.position[0],\
                    relevant_position[1] - self.position[1])
            if (boid != self and
                    vector.magnitude(*diff) <= self.viewing_range and\
                    vector.angle_between(self.velocity, diff) <=\
                    self.viewing_angle):
                yield boid
        return


    def average_position(self, nearby_boids):
        '''
        take the average position of all nearby boids, and move the boid towards that point
        '''
        if len(nearby_boids) > 0:
            sum_x, sum_y = 0.0, 0.0
            for boid in nearby_boids:
                relevant_position = vector.relevant_position(self.position,\
                                                    boid.position, self.bounds)
                sum_x += relevant_position[0]
                sum_y += relevant_position[1]

            average_x, average_y = (sum_x / len(nearby_boids), sum_y /\
                                    len(nearby_boids))
            return [average_x -self.position[0], average_y - self.position[1]]
        else:
            return [0.0, 0.0]


    def average_velocity(self, nearby_boids):
        '''
        take the average velocity of all nearby boids
        '''
        if len(nearby_boids) > 0:
            sum_x, sum_y = 0.0, 0.0
            for boid in nearby_boids:
                sum_x += boid.velocity[0]
                sum_y += boid.velocity[1]

            average_x, average_y = (sum_x / len(nearby_boids), sum_y /\
                                    len(nearby_boids))
            return [average_x - self.velocity[0], average_y - self.velocity[1]]
        else:
            return [0.0, 0.0]


    def determine_nearby_objects(self, all_objects, collision_distance):
        '''
        return a list of all nearby objects (including objects and predators)
        that are closer than the collision distance
        '''
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
        '''
        move away from nearby objects and predators
        '''
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
        return c

    def attraction(self, attractors):
        '''
        move towards attractors
        '''
        a = [0.0, 0.0]
        if not attractors:
            return a

        for attractor in attractors:
            relevant_position = vector.relevant_position(self.position,\
                                                attractor.position, self.bounds)
            a[0] += relevant_position[0] - self.position[0]
            a[1] += relevant_position[1] - self.position[1]

        return a


    def update(self, dt, all_boids, attractors, obstacles, predators):

        nearby_boids = list(self.determine_nearby_boids(all_boids))

        # update the boid's direction based on several behavioural rules
        cohesion_vector = self.average_position(nearby_boids)
        alignment_vector = self.average_velocity(nearby_boids)
        attractor_vector = self.attraction(attractors)
        boid_avoidance_vector = self.avoid_collisions(all_boids,\
                                                    _BOID_COLLISION_DISTANCE)
        obstacle_avoidance_vector = self.avoid_collisions(obstacles,\
                                                _OBSTACLE_COLLISION_DISTANCE)
        predator_avoidance_vector = self.avoid_collisions(predators,\
                                                _PREDATOR_COLLISION_DISTANCE)

        self.change_vectors = [
            (_COHESION_FACTOR, cohesion_vector),
            (_ALIGNMENT_FACTOR, alignment_vector),
            (_ATTRACTOR_FACTOR, attractor_vector),
            (_BOID_AVOIDANCE_FACTOR, boid_avoidance_vector),
            (_OBSTACLE_AVOIDANCE_FACTOR, obstacle_avoidance_vector),
            (_PREDATOR_COLLISION_FACTOR, predator_avoidance_vector)]

        for factor, vec in self.change_vectors:
            self.velocity[0] += factor *vec[0]
            self.velocity[1] += factor *vec[1]

        # ensure that the boid's velocity is <= self.max_speed
        self.velocity = vector.limit_magnitude(self.velocity, self.max_speed,\
                                                _MIN_SPEED)

        # move the boid to its new position, given its current velocity and the boundary constraints
        for i in range(0, len(self.position)):
            self.position[i] += dt * self.velocity[i]
            if self.position[i] >= self.wrap_bounds[i]:
                self.position[i] = (self.position[i] % self.wrap_bounds[i]) -\
                                                        _BOUNDARY_SLOP
            elif self.position[i] < -_BOUNDARY_SLOP:
                self.position[i] = self.position[i] + self.wrap_bounds[i] +\
                                                        _BOUNDARY_SLOP
