#!/bin/python3

from math import floor

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, Sum, gamma, Piecewise

nlp = Nlp('pinene')

nh = S(400) # number of partition intervals
IntervalNumber = IdxType('IntervalNumber', (1, nh))
i = IntervalNumber('i')
i1 = IntervalNumber('i1', (IntervalNumber.start, IntervalNumber.end - 1))
si1 = ShiftedIdx(i1, 1)
				    
nc = S(3) # number of collocation points
PointNumber = IdxType('PointNumber', (1, nc))
j = PointNumber('j')
k = PointNumber('k')

# roots of nc-th degree Legendre polynomial
rho = nlp.add_user_data('rho', (j,), init = [0.11270166537926, 0.50000000000000, 0.88729833462074])

ne = S(5) # number of differential equations
EqNumber = IdxType('EqNumber', (1, ne))
s = EqNumber('s')

# times at which observations made
tau = [1230.0, 3060.0, 4920.0, 7800.0, 10680.0, 15030.0, 22620.0, 36420.0]

nm = len(tau) # number of measurements
MeasurementNumber = IdxType('MeasurementNumber', (1, nm))
m = MeasurementNumber('m')


# observations
z = nlp.add_user_data('z_data', (m, s), init = [88.35, 7.3, 2.3, 0.4, 1.75, 76.4, 15.6, 4.5, 0.7, 2.8, 65.1, 23.1, 5.3, 1.1, 5.8, 50.4, 32.9, 6.0, 1.5, 9.3, 37.5, 42.7, 6.0, 1.9, 12.0, 25.9, 49.1, 5.9, 2.2, 17.0, 14.0, 57.4, 5.1, 2.6, 21.0, 4.5, 63.1, 3.8, 2.9, 25.7])

bc = nlp.add_user_data('bc', (s,), init = [100.0, 0.0, 0.0, 0.0, 0.0]) # boundary conditions

tf = tau[nm - 1] # ODEs defined in [0,tf]

h = tf / nh # uniform interval length

fact = lambda j : gamma(j + S.One) 

t = lambda i : (i - 1) * h   # partition

# itau(i) is the largest integer k with t[k] <= tau[i]

itau = lambda i : min(nh, floor(tau[i - 1] / h) + 1);    

np = S(5) # number of ODE parameters
ParamNumber = IdxType('ParamNumber', (1, np))
l = ParamNumber('l')
theta = nlp.add_var('theta', (l,), starting_point = S.Zero, lower = S.Zero) #  ODE parameters

# The collocation approximation u is defined by the parameters v and w.
# uc and Duc are, respectively, u and u' evaluated at the collocation points.

cond = [(bc[s], i <= itau(1))]
for m in range(2, nm + 1) : 
  cond.append((z[MeasurementNumber(m), s], i <= itau(m)))
if itau(nm) + 1 <= nh :
  cond.append((z[MeasurementNumber(nm), s], True))
v = nlp.add_var('v', (i, s), starting_point = Piecewise(*cond))
w = nlp.add_var('w', (i, j, s), starting_point = S.Zero) #!!!!

uc = lambda i, j, s : v[i, s] + h * Sum(w[i, k, s] * (rho[j]**k / fact(k)), k)

Duc = lambda i, j, s : Sum(w[i, k, s] * (rho[j]**(k - 1) / fact(k - 1)), k)

f = S.Zero
for m in MeasurementNumber.range() :
  f += Sum((v[IntervalNumber(itau(m)), s] + Sum(w[IntervalNumber(itau(m)), k, s] * (tau[m - 1] - t(itau(m)))**k / (fact(k) * h**(k - 1)), k) - z[MeasurementNumber(m), s])**2, s)

nlp.set_obj(f)

nlp.add_constr(v[IntervalNumber(1), s], lower = bc[s], upper = bc[s])

nlp.add_constr(v[si1, s] - v[i1, s] - h * Sum(w[i1, k, s] / fact(k), k), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(1))  + (theta[ParamNumber(1)] + theta[ParamNumber(2)]) * uc(i, j, EqNumber(1)), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(2)) - theta[ParamNumber(1)] * uc(i, j, EqNumber(1)), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(3)) - theta[ParamNumber(2)] * uc(i, j, EqNumber(1)) + (theta[ParamNumber(3)] + theta[ParamNumber(4)])*uc(i, j, EqNumber(3)) - theta[ParamNumber(5)]*uc(i,j, EqNumber(5)), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(4)) - theta[ParamNumber(3)]*uc(i, j, EqNumber(3)), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(5)) - theta[ParamNumber(4)]*uc(i,j,EqNumber(3)) + theta[ParamNumber(5)]*uc(i, j, EqNumber(5)), lower = S.Zero, upper = S.Zero)

nlp.generate()
