#!/bin/python3

from sympy import S, Symbol
from sympy2ipopt import CauchyProblem, IdxType, Nlp, ShiftedIdx


nlp = Nlp('cauchy_linear')

TimeMesh = IdxType('TimeMesh', (0, 10))
i = TimeMesh('i')
k = TimeMesh('k', (TimeMesh.start, TimeMesh.end - 1))
sk = ShiftedIdx(k, 1)

t = Symbol('t', real=True)
x_sym = Symbol('x', real=True)
p_sym = Symbol('p', real=True)

x = nlp.add_var('x', (i,), starting_point=S.One)
p = nlp.add_var('p', starting_point=S.One)

ode = CauchyProblem(
  nlp,
  'linear_cauchy',
  time=t,
  state=(x_sym,),
  rhs=(p_sym * x_sym,),
  parameters=(p_sym,),
  include_time_bounds=False,
)

phi = ode.solution(
  initial_state=(x[k],),
  parameters=(p,),
)

nlp.set_obj(x[TimeMesh(TimeMesh.end)])
nlp.add_constr(x[TimeMesh(TimeMesh.start)], lower=S.One, upper=S.One)
nlp.add_constr(x[sk] - phi[0], lower=S.Zero, upper=S.Zero)

nlp.generate()
