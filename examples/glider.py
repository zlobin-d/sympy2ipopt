#!/bin/python3

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, pi, sqrt, exp

nlp = Nlp('glider')

x_0 = S.Zero
y_0 = S(1000)
y_f = S(900)

vx_0 = S(13.23)
vx_f = S(13.23)

vy_0 = S(-1.288)
vy_f = S(-1.288)

u_c = S(2.5)
r_0 = S(100)

m = S(100)
g = S(9.81)
c0 = S(0.034)
c1 = S(0.069662)
Sq = S(14)
rho = S(1.13)

cL_min = S.Zero
cL_max = S(1.4)

nh = S(400) # Time steps

t_f = nlp.add_var('t_f', starting_point = S.One, lower = S.Zero)

step = t_f / nh;

TimeMesh = IdxType('TimeMesh', (0, nh))
i = TimeMesh('i')
j = TimeMesh('j', (TimeMesh.start + 1, TimeMesh.end))
sj = ShiftedIdx(j, -1)

x = nlp.add_var('x', (i,), starting_point = x_0 + vx_0 * (i / nh), lower = S.Zero) # State variables
y = nlp.add_var('y', (i,), starting_point = y_0 + (i / nh) * (y_f - y_0))
vx = nlp.add_var('vx', (i,), starting_point = vx_0, lower = S.Zero)
vy = nlp.add_var('vy', (i,), starting_point = vy_0)
cL = nlp.add_var('cL', (i,), starting_point = cL_max / 2, lower = cL_min, upper = cL_max) # Control variable

# Functions that define the glider.

r = lambda i : (x[i] / r_0 - 2.5)**2
u = lambda i : u_c * (1 - r(i)) * exp(-r(i))

w = lambda i : vy[i] - u(i)
v = lambda i : sqrt(vx[i]**2 + w(i)**2)

D = lambda i : 0.5 * (c0 + c1 * cL[i]**2) * rho * Sq * v(i)**2
L = lambda i : 0.5 * cL[i] * rho * Sq * v(i)**2

# Functions in the equations of motion.

vx_dot = lambda i : (-L(i) * (w(i) / v(i)) - D(i) * (vx[i] / v(i))) / m
vy_dot = lambda i : (L(i) * (vx[i] / v(i)) - D(i) * (w(i) / v(i))) / m - g

nlp.set_obj(-x[TimeMesh(TimeMesh.end)])

nlp.add_constr(x[j] - x[sj] - 0.5 * step * (vx[j] + vx[sj]), lower = S.Zero, upper = S.Zero)
nlp.add_constr(y[j] - y[sj] - 0.5 * step * (vy[j] + vy[sj]), lower = S.Zero, upper = S.Zero)
nlp.add_constr(vx[j] - vx[sj] - 0.5 * step * (vx_dot(j) + vx_dot(sj)), lower = S.Zero, upper = S.Zero)
nlp.add_constr(vy[j] - vy[sj] - 0.5 * step * (vy_dot(j) + vy_dot(sj)), lower = S.Zero, upper = S.Zero)

# Boundary Conditions

nlp.add_constr(x[TimeMesh(TimeMesh.start)], lower = x_0, upper = x_0)
nlp.add_constr(y[TimeMesh(TimeMesh.start)], lower = y_0, upper = y_0)
nlp.add_constr(y[TimeMesh(TimeMesh.end)], lower = y_f, upper = y_f)
nlp.add_constr(vx[TimeMesh(TimeMesh.start)], lower = vx_0, upper = vx_0)
nlp.add_constr(vx[TimeMesh(TimeMesh.end)], lower = vx_f, upper = vx_f)
nlp.add_constr(vy[TimeMesh(TimeMesh.start)], lower = vy_0, upper = vy_0)
nlp.add_constr(vy[TimeMesh(TimeMesh.end)], lower = vy_f, upper = vy_f)

nlp.generate()
