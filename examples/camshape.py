#!/bin/python3

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, pi, cos, Sum

nlp = Nlp('camshape')

n = S(1200) # number of discretization points 
R_v = S.One # design parameter related to the valve shape
R_max = S(2.0) # maximum allowed radius of the cam
R_min = S.One # minimum allowed radius of the cam
alpha = S(1.5) # # curvature limit parameter
d_theta = 2*pi/(5*(n + 1)) # angle between discretization points

TimeMesh = IdxType('TimeMesh', (1, n))
i = TimeMesh('i')
i1 = TimeMesh('i', (TimeMesh.start + 1, TimeMesh.end - 1))
sli1 = ShiftedIdx(i1, -1) 
sri1 = ShiftedIdx(i1, 1) 
i2 = TimeMesh('i', (TimeMesh.start, TimeMesh.end - 1))
sri2 = ShiftedIdx(i2, 1)

# radius of the cam at discretization points
r = nlp.add_var('r', (i,), starting_point = (R_min + R_max) / 2, lower = R_min, upper = R_max) 


nlp.set_obj((-pi*R_v)/n * Sum(r[i], i))

nlp.add_constr(-r[sli1]*r[i1] - r[i1]*r[sri1] + 2*r[sli1]*r[sri1]*cos(d_theta), upper = S.Zero)

nlp.add_constr(-R_min*r[TimeMesh(1)] - r[TimeMesh(1)]*r[TimeMesh(2)] + 2*R_min*r[TimeMesh(2)]*cos(d_theta), upper = S.Zero)
nlp.add_constr(-R_min**2 - R_min*r[TimeMesh(1)] + 2*R_min*r[TimeMesh(1)]*cos(d_theta), upper = S.Zero)
nlp.add_constr(-r[TimeMesh(n - 1)]*r[TimeMesh(n)] - r[TimeMesh(n)]*R_max + 2*r[TimeMesh(n-1)]*R_max*cos(d_theta), upper = S.Zero)
nlp.add_constr(-2*R_max*r[TimeMesh(n)] + 2*r[TimeMesh(n)]**2*cos(d_theta), upper = S.Zero)


nlp.add_constr(r[sri2] - r[i2], lower = -alpha*d_theta, upper = alpha*d_theta)

nlp.add_constr(r[TimeMesh(1)] - R_min, lower = -alpha*d_theta, upper = alpha*d_theta)
nlp.add_constr(R_max - r[TimeMesh(n)], lower = -alpha*d_theta, upper = alpha*d_theta)

nlp.generate()