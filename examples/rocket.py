#!/bin/python3

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, pi, sin, sqrt, exp

nlp = Nlp('rocket')

nh = S(800)  # Number of intervals in mesh

# Parameters for the model.

T_c = S(3.5)
h_c = S(500)
v_c = S(620)
m_c = S(0.6)

v_0 = S.Zero # Initial velocity

# Normalization of the equations allows g_0 = h_0 = m_0 = 1

h_0 = S.One # Initial height
m_0 = S.One # Initial mass
g_0 = S.One # Gravity at the surface

# Derived parameters.

c = 0.5 * sqrt(g_0 * h_0)
m_f = m_c * m_0
D_c = 0.5 * v_c * (m_0 / g_0)
T_max = T_c * (m_0 * g_0)

# Height, velocity, mass and thrust of rocket.

TimeMesh = IdxType('TimeMesh', (0, nh))
i = TimeMesh('i')
j = TimeMesh('j', (TimeMesh.start + 1, TimeMesh.end))
sj = ShiftedIdx(j, -1)

h = nlp.add_var('h', (i,), starting_point = S.One, lower = h_0)
v = nlp.add_var('v', (i,), starting_point = (i / nh) * (1 - (i / nh)), lower = S.Zero)
m = nlp.add_var('m', (i,), starting_point = (m_f - m_0) * (i / nh) + m_0, lower = m_f, upper = m_0)
T = nlp.add_var('T', (i,), starting_point = T_max / 2, lower = S.Zero, upper = T_max)
step = nlp.add_var('step', starting_point = 1 / nh, lower = S.Zero)

# Drag function.

D = lambda i : D_c * (v[i]**2) * exp(-h_c * (h[i] - h_0) / h_0)

# Gravity function.

g = lambda i : g_0 * (h_0 / h[i])**2

nlp.set_obj(h[TimeMesh(TimeMesh.end)])

nlp.add_constr(h[j] - h[sj] - 0.5 * step * (v[j] + v[sj]), lower = S.Zero, upper = S.Zero)

nlp.add_constr(v[j] - v[sj] - 0.5 * step * ((T[j] - D(j) - m[j] * g(j)) / m[j] - (T[sj] - D(sj) - m[sj] * g(sj)) / m[sj]), lower = S.Zero, upper = S.Zero)

nlp.add_constr(m[j] - m[sj] + 0.5 * step * (T[j] + T[sj]) / c, lower = S.Zero, upper = S.Zero)

# Boundary Conditions

nlp.add_constr(h[TimeMesh(TimeMesh.start)], lower = h_0, upper = h_0)
nlp.add_constr(v[TimeMesh(TimeMesh.start)], lower = v_0, upper = v_0)
nlp.add_constr(m[TimeMesh(TimeMesh.start)], lower = m_0, upper = m_0)
nlp.add_constr(m[TimeMesh(TimeMesh.end)], lower = m_f, upper = m_f)

nlp.generate()
