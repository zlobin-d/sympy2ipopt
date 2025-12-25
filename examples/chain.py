#!/bin/python3

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, Abs, sqrt

nlp = Nlp('chain')

nh = S(800) # number of subintervals
L = S(4) # length of the suspended chain
a = S(1) # height of the chain at t=0 (left)
b = S(3) # height of the chain at t=1 (right)
tf = S(1.0) # ODEs defined in [0,tf]
h = tf/nh # uniform interval length
tmin = S(0.25) if b > a else S(0.75)

TimeMesh = IdxType('TimeMesh', (0, nh))
i = TimeMesh('i')
j = TimeMesh('j', (TimeMesh.start, TimeMesh.end - 1))
sj = ShiftedIdx(j, 1)

# Set starting point
u = nlp.add_var('u', (i,), starting_point = 4 * Abs(b - a)*((i/nh) - tmin))
x1 = nlp.add_var('x1', (i,), starting_point = 4 * Abs(b - a)*(i/nh)*(0.5*(i/nh) - tmin) + a) # height of the chain
x2 = nlp.add_var('x2', (i,), starting_point = (4 * Abs(b - a)*((i/nh) - tmin)) * (4 * Abs(b - a)*(i/nh)*(0.5*(i/nh) - tmin) + a)) # potential energy of the chain
x3 = nlp.add_var('x3', (i,), starting_point = 4 * Abs(b - a)*((i/nh) - tmin)) # lenght of the chain


nlp.set_obj(x2[TimeMesh(TimeMesh.end)])


nlp.add_constr(x1[sj] - x1[j] - 0.5*h*(u[j] + u[sj]), lower = S.Zero, upper = S.Zero)

nlp.add_constr(x2[sj] - x2[j] - 0.5*h*(x1[j]*sqrt(1 + u[j]**2) + x1[sj]*sqrt(1 + u[sj]**2)), lower = S.Zero, upper = S.Zero)

nlp.add_constr(x3[sj] - x3[j] - 0.5*h*(sqrt(1+u[j]**2) + sqrt(1+u[sj]**2)), lower = S.Zero, upper = S.Zero)

# Boundary conditions
nlp.add_constr(x1[TimeMesh(TimeMesh.start)], lower = a, upper = a)
nlp.add_constr(x1[TimeMesh(TimeMesh.end)], lower = b, upper = b)
nlp.add_constr(x2[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(x3[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(x3[TimeMesh(TimeMesh.end)], lower = L, upper = L)

nlp.generate()