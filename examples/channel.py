#!/bin/python3

from math import floor

from sympy2ipopt import Nlp, IdxType, ShiftedIdx

from sympy import S, Sum, gamma, Piecewise

nlp = Nlp('channel')


nh = S(800) # number of partition intervals
IntervalNumber = IdxType('IntervalNumber', (1, nh))
i = IntervalNumber('i')
i1 = IntervalNumber('i1', (IntervalNumber.start, IntervalNumber.end - 1))
si1 = ShiftedIdx(i1, 1)
				    
nc = S(4) # number of collocation points
PointNumber = IdxType('PointNumber', (1, nc))
j = PointNumber('j')
k = PointNumber('k')


nd = S(4) # order of the differential equation
DIfEqNumber = IdxType('DifEqNumber', (1, nd))
s = DIfEqNumber('s')

# roots of nc-th degree Legendre polynomial
rho = nlp.add_user_data('rho', (j,), init = [0.06943184420297, 0.33000947820757, 0.66999052179243, 0.93056815579703])


bc = nlp.add_user_data('bc', (s,), init = [0.0, 1.0, 0.0, 0.0]) # boundary conditions

tf = S.One # ODEs defined in [0,tf]

h = tf / nh # uniform interval length

t = lambda i : (i - 1) * h   # partition

fact = lambda j : gamma(j + S.One) # factorial function

R = S(10) # Reynolds number


v1 = lambda i: t(i)**2*(3 - 2*t(i))
v2 = lambda i: 6*t(i)*(1-t(i))
v3 = lambda i: 6*(1.0 - 2*t(i))
v4 = lambda i: -12



w = nlp.add_var('w', (i, j), starting_point = S.Zero) #!!!!

uc1 = lambda i, j : v1(i) + h * Sum(w[i, k] * (rho[j]**k / fact(k)), k)
uc2 = lambda i, j : v2(i) + h * Sum(w[i, k] * (rho[j]**k / fact(k)), k)
uc3 = lambda i, j : v3(i) + h * Sum(w[i, k] * (rho[j]**k / fact(k)), k)
uc4 = lambda i, j : v4(i) + h * Sum(w[i, k] * (rho[j]**k / fact(k)), k)

Duc1 = lambda i, j : v1(i)*((rho[j]*h)**0/fact(0)) + v2(i)*((rho[j]*h)**1/fact(1)) + v3(i)*((rho[j]*h)**2/fact(2)) + v4(i)*((rho[j]*h)**3/fact(3)) + h**(nd-1+1)*Sum(w[i,k]*(rho[j]**(k+nd-1)/fact(k+nd-1)), k)
Duc2 = lambda i, j : v2(i)*((rho[j]*h)**0/fact(0)) + v3(i)*((rho[j]*h)**1/fact(1)) + v4(i)*((rho[j]*h)**2/fact(2)) + h**(nd-1+1)*Sum(w[i,k]*(rho[j]**(k+nd-1)/fact(k+nd-1)), k)
Duc3 = lambda i, j : v3(i)*((rho[j]*h)**0/fact(0)) + v4(i)*((rho[j]*h)**1/fact(1)) + h**(nd-1+1)*Sum(w[i,k]*(rho[j]**(k+nd-1)/fact(k+nd-1)), k)
Duc4 = lambda i, j : v4(i)*((rho[j]*h)**0/fact(0)) + h**(nd-1+1)*Sum(w[i,k]*(rho[j]**(k+nd-1)/fact(k+nd-1)), k)

constraint_objective = nlp.add_var('constraint_objective', starting_point=S.One, lower=S.One, upper=S.One)

nlp.set_obj(constraint_objective)


nlp.add_constr(v1(1) - bc[DIfEqNumber(DIfEqNumber.start)], lower = S.Zero, upper = S.Zero)
nlp.add_constr(v2(1) - bc[DIfEqNumber(DIfEqNumber.start + 1)], lower = S.Zero, upper = S.Zero)

nlp.add_constr(v1(nh)*(h**(0)/fact(0)) + v2(nh)*(h**(1)/fact(1)) + v3(nh)*(h**(2)/fact(2)) + v4(nh)*(h**(3)/fact(3)) + h**nd*Sum(w[IntervalNumber(IntervalNumber.end), k]/fact(k+nd-1), k) - bc[DIfEqNumber(DIfEqNumber.start + 2)], lower = S.Zero, upper = S.Zero)

nlp.add_constr(v2(nh)*(h**(0)/fact(0)) + v3(nh)*(h**(1)/fact(1)) + v4(nh)*(h**(2)/fact(2)) + h**(nd-1)*Sum(w[IntervalNumber(IntervalNumber.end), k]/fact(k+nd-2), k) - bc[DIfEqNumber(DIfEqNumber.start + 3)], lower = S.Zero, upper = S.Zero)

nlp.add_constr(v1(i1)*(h**0/fact(0)) + v2(i1)*(h**1/fact(1)) + v3(i1)*(h**2/fact(2)) + v4(i1)*(h**3/fact(3)) + h**(nd-1+1)*Sum(w[i1, k]/fact(k+nd-1), k) - v1(si1), lower = S.Zero, upper = S.Zero)
nlp.add_constr(v2(i1)*(h**0/fact(0)) + v3(i1)*(h**1/fact(1)) + v4(i1)*(h**2/fact(2)) + h**(nd-2+1)*Sum(w[i1, k]/fact(k+nd-2), k) - v2(si1), lower = S.Zero, upper = S.Zero)
nlp.add_constr(v3(i1)*(h**0/fact(0)) + v4(i1)*(h**1/fact(1)) + h**(nd-3+1)*Sum(w[i1, k]/fact(k+nd-3), k) - v3(si1), lower = S.Zero, upper = S.Zero)
nlp.add_constr(v4(i1)*(h**0/fact(0)) + h**(nd-4+1)*Sum(w[i1, k]/fact(k+nd-4), k) - v4(si1), lower = S.Zero, upper = S.Zero)

nlp.add_constr(Sum(w[i,k]*(rho[j]**(k-1)/fact(k-1)), k) - R*(Duc2(i, j)*Duc3(i, j) - Duc1(i, j)*Duc4(i, j)), lower = S.Zero, upper = S.Zero)

nlp.generate()
