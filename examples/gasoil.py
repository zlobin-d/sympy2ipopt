#!/bin/python3

from math import floor

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, Sum, gamma, Piecewise

nlp = Nlp('gasoil')

nh = 400 # number of partition intervals
IntervalNumber = IdxType('IntervalNumber', (1, nh))
i = IntervalNumber('i')
i1 = IntervalNumber('i1', (IntervalNumber.start, IntervalNumber.end - 1))
si1 = ShiftedIdx(i1, 1)
				    
nc = S(4) # number of collocation points
PointNumber = IdxType('PointNumber', (1, nc))
j = PointNumber('j')
k = PointNumber('k')

# roots of nc-th degree Legendre polynomial
rho = nlp.add_user_data('rho', (j,), init = [0.06943184420297, 0.33000947820757, 0.66999052179243, 0.93056815579703])

ne = S(2) # number of differential equations
EqNumber = IdxType('EqNumber', (1, 2))
s = EqNumber('s')

# times at which observations made
tau = [0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.150, 0.175, 0.20, 0.225, 0.250, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95]

nm = len(tau) # number of measurements
MeasurementNumber = IdxType('MeasurementNumber', (1, nm))
m = MeasurementNumber('m')

# observations
z = nlp.add_user_data('z_data', (m, s), init = [1.0000, 0, 0.8105, 0.2000, 0.6208, 0.2886, 0.5258, 0.3010, 0.4345, 0.3215, 0.3903, 0.3123, 0.3342, 0.2716, 0.3034, 0.2551, 0.2735, 0.2258, 0.2405, 0.1959, 0.2283, 0.1789, 0.2071, 0.1457, 0.1669, 0.1198, 0.1530, 0.0909, 0.1339, 0.0719, 0.1265, 0.0561, 0.1200, 0.0460, 0.0990, 0.0280, 0.0870, 0.0190, 0.0770, 0.0140, 0.0690, 0.0100])

bc = nlp.add_user_data('bc', (s,), init = [1.0, 0.0]) # ODE initial conditions

tf = tau[nm - 1] # ODEs defined in [0,tf]

h = tf / nh # uniform interval length

fact = lambda j : gamma(j + S.One) 
#fact = nlp.add_user_func('fact', 1)

t = lambda i : (i - 1) * h   # partition

# itau(i) is the largest integer k with t[k] <= tau[i]

itau = lambda i : min(nh, floor(tau[i - 1] / h) + 1);    

np = S(3) # number of ODE parameters
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
w = nlp.add_var('w', (i, j, s), starting_point = S.One)

uc = lambda i, j, s : v[i, s] + h * Sum(w[i, k, s] * (rho[j]**k / fact(k)), k)

Duc = lambda i, j, s : Sum(w[i, k, s] * (rho[j]**(k - 1) / fact(k - 1)), k)

f = S.Zero
for m in MeasurementNumber.range() :
  f += Sum((v[IntervalNumber(itau(m)), s] + Sum(w[IntervalNumber(itau(m)), k, s] * (tau[m - 1] - t(itau(m)))**k / (fact(k) * h**(k - 1)), k) - z[MeasurementNumber(m), s])**2, s)

nlp.set_obj(f)

nlp.add_constr(v[IntervalNumber(1), s], lower = bc[s], upper = bc[s])

nlp.add_constr(v[si1, s] - v[i1, s] - Sum(w[i1, k, s] * h / fact(k), k), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(1))  + (theta[ParamNumber(1)] + theta[ParamNumber(3)]) * uc(i, j, EqNumber(1))**2, lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(2)) - theta[ParamNumber(1)] * uc(i, j, EqNumber(1))**2 + theta[ParamNumber(2)] * uc(i, j, EqNumber(2)), lower = S.Zero, upper = S.Zero)

nlp.generate()
