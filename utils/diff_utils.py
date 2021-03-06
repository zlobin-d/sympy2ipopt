#!/bin/python3

from collections import defaultdict
from functools import reduce
from itertools import product, starmap
from sympy import Sum, diff, S, Function, Derivative, Dummy, Symbol
from sympy2ipopt.shifted_idx import ShiftedIdx
from sympy2ipopt.utils.idx_utils import get_master_idx, get_shifts, get_types, block_copy, idx_subs
from sympy2ipopt.utils.expr_utils import RDummy, wrap_in_sum, prepare_expr

def get_sum_with_indexed(expr, var) :
  ''' Находим Sum, в которых суммируется var.

  Пользуемся уникальностью индексов суммирования:

  #. переменная, при которой есть некоторый индекс, не может суммироваться по нему в двух разных суммах
  #. это точно индекс суммирования, он не может совпасть с внешним индексом
  '''

  var_indices_set = set(get_master_idx(var.indices))
  def get(expr) :
    if isinstance(expr, Sum) :
      term, *indices = expr.args
      if any(map(lambda idx : idx[0] in var_indices_set, indices)) and term.has(var) :
        return expr
    for arg in expr.args :
      ret = get(arg)
      if ret != None :
        return ret
    return None
  return get(expr)

def diff_indexed(expr, var, occurrences) :
  '''Вычисляем производную выражения по блочной переменной.'''

  def partial(expr, var, occur) :
    assert var.base == occur.base
    sum_with_var = get_sum_with_indexed(expr, occur)
    service_var = RDummy()
    if sum_with_var == None :
      return diff(expr.subs(occur, service_var), service_var).subs(service_var, occur)
    else :
      service_func = Function(RDummy())(service_var)
      expr = expr.xreplace({sum_with_var : service_func})
      expr = diff(expr, service_var)
      term, *sum_indices = sum_with_var.args
      sum_indices = {idx[0] for idx in sum_indices}
      d_term = partial(term, var, occur)
      for old, new in zip(occur.indices, var.indices) :
        old_master, = get_master_idx((old,))
        if old_master in sum_indices :
          d_term = idx_subs(d_term, old, new)
          sum_indices.remove(old_master)
      d_term = wrap_in_sum(d_term, sum_indices)
      expr = expr.subs(Derivative(service_func, service_var), d_term)
      return expr.subs(service_func, sum_with_var)
  derivative = S.Zero
  for occur in occurrences :
    derivative += partial(expr, var, occur)
  # сохраняем условие уникальности индексов
  return prepare_expr(derivative)

if __name__ == "__main__" :
  from sympy import cos, sin, Idx, exp
  from sympy2ipopt.idx_type import IdxType
  from sympy2ipopt.indexed_base_with_offset import IndexedBaseWithOffset
  from sympy2ipopt.utils.idx_utils import IDummy
  from sympy2ipopt.utils.test_utils import renum_dummy

  t1 = IdxType('t1', (0, 10))
  t2 = IdxType('t2', (2,8))
  t3 = IdxType('t3', (-5,3))
  t4 = IdxType('t4', (-9,-4))
  b1 = IndexedBaseWithOffset('b1')
  b2 = IndexedBaseWithOffset('b2')

  i1 = t1('i1')
  i2 = t2('i2')
  i3 = t3('i3')
  i4 = t4('i4')

  j1 = t1('j1', (0, 9))
  j2 = t2('j2', (5, 7))
  j3 = t3('j3', (-1, 1))
  j4 = t4('j4', (-8, -6))
  sj1 = ShiftedIdx(j1, 1)
  sj2 = ShiftedIdx(j2, -2)
  sj3 = ShiftedIdx(j3, -3)
  sj4 = ShiftedIdx(j4, 2)

  k2 = t2('k2', (6, 6))
  k3 = t3('k3', (0, 0))

  l1= t1('l1', (10, 10))
  l2= t1('l2', (2, 4))

  r1 = IndexedBaseWithOffset('r', (1,), 3)
  r2 = IndexedBaseWithOffset('r', (1,), 5)

  sum1 = Sum(b1[i1, k3], i1, k3)
  sum2 = Sum(b2[sj2, sj3], j2)
  sum3 = Sum(r1[i1] + sum1, i1)
  sum4 = Sum(b1[i1, j2] + b2[i3, i4], i3, i4)
  sum5 = Sum(b2[i1, i4], i1)

  assert get_sum_with_indexed(sum1, b1[i1, k3]) == sum1
  assert get_sum_with_indexed(5 * sum2, b2[sj2, sj3]) == sum2
  assert get_sum_with_indexed(sum4 + 8, b1[i1, j2]) == None
  assert get_sum_with_indexed(cos(sum4) + sin(sum1)**3 , b2[i3, i4]) == sum4
  assert get_sum_with_indexed(5 * sum5**sum4, b2[i1, i4]) == sum5

  assert diff_indexed(b1[i1, i2]**2, b1[i1, i2], (b1[i1, i2],)) == 2 * b1[i1, i2]
  assert diff_indexed(b1[j1, j2]**2, b1[l1, l2], (b1[l1, l2],)) == S.Zero
  assert diff_indexed(b1[j1, j2]**2 * b2[i3], b2[i3], (b2[i3],)) == b1[j1, j2]**2
  assert diff_indexed(b1[j1, j2]**2 + b1[l1, l2], b1[l1, l2], (b1[l1, l2],)) == S.One
  assert diff_indexed(sum4, b1[i1, j2], (b1[i1, j2],)).doit() == Sum(S.One, i3, i4).doit()
  assert diff_indexed(b1[j1, j2] + sum5, b1[j1, j2], (b1[j1, j2],)).doit() == S.One
  sum6 = Sum(b1[i1]**2 * b2[i2], i2)
  assert diff_indexed(sum6 + sin(b1[i1]), b1[i1], (b1[i1],)) == Sum(2 * b1[i1] * b2[i2], i2) + cos(b1[i1])
  sum7 = Sum(b1[i1]**2 * b2[i1], i1)
  assert diff_indexed(sum7, b1[i1], (b1[i1],)) == 2 * b1[i1] * b2[i1]
  sum8 = Sum(b1[i1, j2, sj3, sj4]**2, i1, j2, j3)
  assert diff_indexed(sum8, b1[i1, j2, sj3, sj4], (b1[i1, j2, sj3, sj4],)) == 2 * b1[i1, j2, sj3, sj4]
  assert renum_dummy(diff_indexed(exp(sum8), b1[i1, j2, sj3, sj4], (b1[i1, j2, sj3, sj4],))) == renum_dummy(prepare_expr(exp(sum8) * 2 * b1[i1, j2, sj3, sj4]))
  sum9 = Sum(b1[i1, j2, sj3, sj4]**2 + b2[l2], i1, j2, j3, l2)
  assert diff_indexed(sum9, b1[i1, j2, sj3, sj4], (b1[i1, j2, sj3, sj4],)) == 6 * b1[i1, j2, sj3, sj4]
  m0 = t1('m0', (0, 0))
  sm0 = ShiftedIdx(m0, 1)
  m1 = t1('m1', (1, 9))
  sm1_0 = ShiftedIdx(m1, -1)
  sm1_1 = ShiftedIdx(m1, 1)
  m2 = t1('m2', (10, 10))
  sm2 = ShiftedIdx(m2, -1)
  assert diff_indexed(Sum((b1[j1] - b1[sj1])**2, j1), b1[m0], (b1[j1],)) == -2*b1[sm0] + 2*b1[m0]
  assert diff_indexed(Sum((b1[j1] - b1[sj1])**2, j1), b1[m1], (b1[j1], b1[sj1])) == -2*b1[sm1_1] - 2*b1[sm1_0] + 4*b1[m1]
  assert diff_indexed(Sum((b1[j1] - b1[sj1])**2, j1), b1[m2], (b1[sj1],)) == -2*b1[sm2] + 2*b1[m2]
  assert renum_dummy(diff_indexed(Sum(i2*(b1[j1] - b1[sj1])**2, j1, i2), b1[m1], (b1[j1], b1[sj1]))) == renum_dummy(prepare_expr(Sum((-2*b1[sm1_1] + 2*b1[m1]) * i2, i2) + Sum((-2*b1[sm1_0] + 2*b1[m1]) * i2, i2)))
  n1 = t1('n1', (3, 5))
  assert diff_indexed(Sum(b1[sj1]**2, j1) + Sum(b1[m1]**2, m1), b1[n1], (b1[sj1], b1[m1])) == 4 * b1[n1]
  n2 = t2('n2', (3, 5))
  assert diff_indexed(Sum(b1[j1] * Sum(b2[j1, sj2]**2, j2), j1), b2[m1, n2], (b2[j1, sj2],)) == 2 * b1[m1] * b2[m1, n2]
  assert diff_indexed(Sum(b1[j1] * Sum(b2[sj1, sj2]**2, j2), j1), b1[m1], (b1[j1],)) == Sum(b2[sm1_1, sj2]**2, j2)
  assert diff_indexed(Sum(j1 * Sum(b1[j1]**2 * b2[sj1, sj2]**2, j2), j1), b1[m1], (b1[j1],)) == m1 * Sum(2 * b1[m1] * b2[sm1_1, sj2]**2, j2)
  assert diff_indexed(Sum(j1 * Sum(b1[j1]**2 * b2[sj1, sj2]**2, j2), j1), b1[m2], (b1[l1],)) == S.Zero
  assert diff_indexed(Sum(j1 * Sum(b1[j1]**2 * b2[sj1, sj2]**2, j2), j1), b1[m1], (b1[j1], b1[i1])) == m1 * Sum(2 * b1[m1] * b2[sm1_1, sj2]**2, j2)

  print('ALL TESTS HAVE BEEN PASSED!!!')

