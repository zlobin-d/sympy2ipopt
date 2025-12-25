#!/bin/python3

from math import floor

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, Sum, gamma, Piecewise

nlp = Nlp('marine')

nh = 400 # number of partition intervals
IntervalNumber = IdxType('IntervalNumber', (1, nh))
i = IntervalNumber('i')
i1 = IntervalNumber('i1', (IntervalNumber.start, IntervalNumber.end - 1))
si1 = ShiftedIdx(i1, 1)
				    
nc = S(1) # number of collocation points
PointNumber = IdxType('PointNumber', (1, nc))
j = PointNumber('j')
k = PointNumber('k')

# roots of nc-th degree Legendre polynomial
rho = nlp.add_user_data('rho', (j,), init = [0.50000000000000])

ne = S(8) # number of differential equations
EqNumber = IdxType('EqNumber', (1, ne))
s = EqNumber('s')
s2 = EqNumber('s2', (EqNumber.start + 1, EqNumber.end - 1))
ss2 = ShiftedIdx(s2, -1)

# times at which observations made
tau = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
nm = len(tau) # number of measurements
MeasurementNumber = IdxType('MeasurementNumber', (1, nm))
m = MeasurementNumber('m')

# observations
z = nlp.add_user_data('z_data', (m, s), init = [20000.0, 17000.0, 10000.0, 15000.0, 12000.0, 9000.0, 7000.0, 3000.0, 12445.0, 15411.0, 13040.0, 13338.0, 13484.0, 8426.0, 6615.0, 4022.0, 7705.0, 13074.0, 14623.0, 11976.0, 12453.0, 9272.0, 6891.0, 5020.0, 4664.0, 8579.0, 12434.0, 12603.0, 11738.0, 9710.0, 6821.0, 5722.0, 2977.0, 7053.0, 11219.0, 11340.0, 13665.0, 8534.0, 6242.0, 5695.0, 1769.0, 5054.0, 10065.0, 11232.0, 12112.0, 9600.0, 6647.0, 7034.0, 943.0, 3907.0, 9473.0, 10334.0, 11115.0, 8826.0, 6842.0, 7348.0, 581.0, 2624.0, 7421.0, 10297.0, 12427.0, 8747.0, 7199.0, 7684.0, 355.0, 1744.0, 5369.0, 7748.0, 10057.0, 8698.0, 6542.0, 7410.0, 223.0, 1272.0, 4713.0, 6869.0, 9564.0, 8766.0, 6810.0, 6961.0, 137.0, 821.0, 3451.0, 6050.0, 8671.0, 8291.0, 6827.0, 7525.0, 87.0, 577.0, 2649.0, 5454.0, 8430.0, 7411.0, 6423.0, 8388.0, 49.0, 337.0, 2058.0, 4115.0, 7435.0, 7627.0, 6268.0, 7189.0, 32.0, 228.0, 1440.0, 3790.0, 6474.0, 6658.0, 5859.0, 7467.0, 17.0, 168.0, 1178.0, 3087.0, 6524.0, 5880.0, 5562.0, 7144.0, 11.0, 99.0, 919.0, 2596.0, 5360.0, 5762.0, 4480.0, 7256.0, 7.0, 65.0, 647.0, 1873.0, 4556.0, 5058.0, 4944.0, 7538.0, 4.0, 44.0, 509.0, 1571.0, 4009.0, 4527.0, 4233.0, 6649.0, 2.0, 27.0, 345.0, 1227.0, 3677.0, 4229.0, 3805.0, 6378.0, 1.0, 20.0, 231.0, 934.0, 3197.0, 3695.0, 3159.0, 6454.0, 1.0, 12.0, 198.0, 707.0, 2562.0, 3163.0, 3232.0, 5566.0])

tf = tau[nm - 1] # ODEs defined in [0,tf]

h = tf / nh # uniform interval length

fact = lambda j : gamma(j + S.One) 
#fact = nlp.add_user_func('fact', 1)

t = lambda i : (i - 1) * h   # partition

# itau(i) is the largest integer k with t[k] <= tau[i]

itau = lambda i : min(nh, floor(tau[i - 1] / h) + 1);    

# The collocation approximation u is defined by the parameters v and w.
# uc and Duc are, respectively, u and u' evaluated at the collocation points.
g = nlp.add_var('g', (s,), starting_point = S.Zero, lower = S.Zero) # growth rates
mr = nlp.add_var('mr', (s,), starting_point = S.Zero, lower = S.Zero) # mortality rates

cond = [(z[MeasurementNumber(1), s], i <= itau(1))]
for m in range(2, nm + 1) : 
  cond.append((z[MeasurementNumber(m), s], i <= itau(m)))
if itau(nm) + 1 <= nh :
  cond.append((z[MeasurementNumber(nm), s], True))
v = nlp.add_var('v', (i, s), starting_point = Piecewise(*cond))
w = nlp.add_var('w', (i, j, s), starting_point = S.Zero)

uc = lambda i, j, s : v[i, s] + h * Sum(w[i, k, s] * (rho[j]**k / fact(k)), k)

Duc = lambda i, j, s : Sum(w[i, k, s] * (rho[j]**(k - 1) / fact(k - 1)), k)

f = S.Zero
for m in MeasurementNumber.range() :
  f += Sum((v[IntervalNumber(itau(m)), s] + Sum(w[IntervalNumber(itau(m)), k, s] * (tau[m - 1] - t(itau(m)))**k / (fact(k) * h**(k - 1)), k) - z[MeasurementNumber(m), s])**2, s)

nlp.set_obj(f)

nlp.add_constr(v[si1, s] - v[i1, s] - h*Sum(w[i1, k, s] / fact(k), k), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(1)) + (mr[EqNumber(1)] + g[EqNumber(1)])*uc(i, j, EqNumber(1)), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, s2) + (mr[s2] + g[s2])*uc(i, j, s2) - g[ss2]*uc(i, j, ss2), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Duc(i, j, EqNumber(ne)) + mr[EqNumber(ne)]*uc(i, j, EqNumber(ne)) - g[EqNumber(ne - 1)]*uc(i, j, EqNumber(ne - 1)), lower = S.Zero, upper = S.Zero)

nlp.generate()
