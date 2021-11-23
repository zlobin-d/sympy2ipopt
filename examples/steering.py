#!/bin/python3

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, pi, sin, cos

nlp = Nlp('steering')

nh = S(800) # Number of subintervals
a = S(100) # Magnitude of force.   
u_min = -pi/2 # Bounds on the control
u_max = pi/2

TimeMesh = IdxType('TimeMesh', (0, nh))
i = TimeMesh('i')
j = TimeMesh('j', (TimeMesh.start, TimeMesh.end - 1))
sj = ShiftedIdx(j, 1)

u = nlp.add_var('u', (i,), starting_point = S.Zero, lower = u_min, upper = u_max) # control
x1 = nlp.add_var('x1', (i,), starting_point = S.Zero) # state variables
x2 = nlp.add_var('x2', (i,), starting_point = 5 * i / nh)
x3 = nlp.add_var('x3', (i,), starting_point = 45 * i / nh)
x4 = nlp.add_var('x4', (i,), starting_point = S.Zero)

tf = nlp.add_var('tf', starting_point = S.One, lower = S.Zero) #final time
h = tf / nh;        # step size

nlp.set_obj(tf)

nlp.add_constr(x1[sj] - x1[j] - 0.5 * h * (x3[j] + x3[sj]), lower = S.Zero, upper = S.Zero)
nlp.add_constr(x2[sj] - x2[j] - 0.5 * h * (x4[j] + x4[sj]), lower = S.Zero, upper = S.Zero)

nlp.add_constr(x3[sj] - x3[j] - 0.5 * h * (a * cos(u[j]) + a * cos(u[sj])), lower = S.Zero, upper = S.Zero)
nlp.add_constr(x4[sj] - x4[j] - 0.5 * h * (a * sin(u[j]) + a * sin(u[sj])), lower = S.Zero, upper = S.Zero)

# Boundary conditions
nlp.add_constr(x1[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(x2[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(x3[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(x4[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)

nlp.add_constr(x2[TimeMesh(TimeMesh.end)], lower = S(5), upper = S(5))
nlp.add_constr(x3[TimeMesh(TimeMesh.end)], lower = S(45), upper = S(45))
nlp.add_constr(x4[TimeMesh(TimeMesh.end)], lower = S.Zero, upper = S.Zero)

nlp.generate()
