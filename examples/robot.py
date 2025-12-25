#!/bin/python3

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, pi, sin

nlp = Nlp('robot')

nh = S(800) # number of intervals.
L = S(5) # total length of arm

# Upper bounds on the controls

max_u_rho = S.One
max_u_the = S.One
max_u_phi = S.One

# The length and the angles theta and phi for the robot arm.

TimeMesh = IdxType('TimeMesh', (0, nh))
i = TimeMesh('i')
j = TimeMesh('j', (TimeMesh.start + 1, TimeMesh.end))
sj = ShiftedIdx(j, -1)

rho = nlp.add_var('rho', (i,), starting_point = S(4.5), lower = S.Zero, upper = L)
the = nlp.add_var('the', (i,), starting_point = (2 * pi / 3) * (i / nh)**2, lower = -pi, upper = pi)
phi = nlp.add_var('phi', (i,), starting_point = pi / 4, lower = S.Zero, upper = pi)

# The derivatives of the length and the angles.
    
rho_dot = nlp.add_var('rho_dot', (i,), starting_point = S.Zero)
the_dot = nlp.add_var('the_dot', (i,), starting_point = (4 * pi / 3) * (i / nh))
phi_dot = nlp.add_var('phi_dot', (i,), starting_point = S.Zero)

# The controls.

u_rho = nlp.add_var('u_rho', (i,), starting_point = S.Zero, lower = -max_u_rho, upper = max_u_rho)
u_the = nlp.add_var('u_the', (i,), starting_point = S.Zero, lower = -max_u_the, upper = max_u_the)
u_phi = nlp.add_var('u_phi', (i,), starting_point = S.Zero, lower = -max_u_phi, upper = max_u_phi)

# The step and the final time.

tf = nlp.add_var('tf', starting_point = S.One)
step = tf / nh

# The moments of inertia.

I_the = lambda i : ((L - rho[i])**3 + rho[i]**3) * (sin(phi[i]))**2 / 3.0
I_phi = lambda i : ((L - rho[i])**3 + rho[i]**3) / 3.0

# The robot arm problem.

nlp.set_obj(tf)

nlp.add_constr(rho[j] - rho[sj] - 0.5 * step * (rho_dot[j] + rho_dot[sj]), lower = S.Zero, upper = S.Zero)
nlp.add_constr(the[j] - the[sj] - 0.5 * step * (the_dot[j] + the_dot[sj]), lower = S.Zero, upper = S.Zero)
nlp.add_constr(phi[j] - phi[sj] - 0.5 * step * (phi_dot[j] + phi_dot[sj]), lower = S.Zero, upper = S.Zero)

nlp.add_constr(rho_dot[j] - rho_dot[sj] - 0.5 * step * (u_rho[j] + u_rho[sj]) / L, lower = S.Zero, upper = S.Zero)
nlp.add_constr(the_dot[j] - the_dot[sj] - 0.5 * step * (u_the[j] / I_the(j) + u_the[sj] / I_the(sj)), lower = S.Zero, upper = S.Zero)
nlp.add_constr(phi_dot[j] - phi_dot[sj] - 0.5 * step * (u_phi[j] / I_phi(j) + u_phi[sj] / I_phi(sj)), lower = S.Zero, upper = S.Zero)

# Boundary Conditions

nlp.add_constr(rho[TimeMesh(TimeMesh.start)], lower = S(4.5), upper = S(4.5))
nlp.add_constr(the[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(phi[TimeMesh(TimeMesh.start)], lower = pi / 4, upper = pi / 4)

nlp.add_constr(rho[TimeMesh(TimeMesh.end)], lower = S(4.5), upper = S(4.5))
nlp.add_constr(the[TimeMesh(TimeMesh.end)], lower = 2 * pi / 3, upper = 2 * pi / 3)
nlp.add_constr(phi[TimeMesh(TimeMesh.end)], lower = pi / 4, upper = pi / 4)

nlp.add_constr(rho_dot[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(the_dot[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(phi_dot[TimeMesh(TimeMesh.start)], lower = S.Zero, upper = S.Zero)

nlp.add_constr(rho_dot[TimeMesh(TimeMesh.end)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(the_dot[TimeMesh(TimeMesh.end)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(phi_dot[TimeMesh(TimeMesh.end)], lower = S.Zero, upper = S.Zero)

nlp.generate()
