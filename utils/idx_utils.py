#!/bin/python3

from functools import reduce
from itertools import starmap
from operator import mul as mul_as_func

from sympy import Indexed, S, Range, Dummy
from sympy2ipopt.shifted_idx import ShiftedIdx

def IDummy() :
  return Dummy(integer = True)

def get_indices(var) :
  return getattr(var, 'indices', tuple())

def get_id(var) :
  return var.base if isinstance(var, Indexed) else var

def get_master_idx(indices) :
  return tuple(getattr(idx, 'idx', idx) for idx in indices)

def get_shifts(indices) :
  return tuple(getattr(idx, 'shift', S.Zero) for idx in indices)

def get_types(indices) :
  return tuple(map(type, get_master_idx(indices)))

def is_full_range(indices) :
  return all(map(lambda idx : len(idx) == len(type(idx)), indices))

def get_idx_pos(indices, indices_set) :
  return tuple(n for n, idx in enumerate(indices) if idx in indices_set)

def can_be_eq(idx1, idx2, outer1 = set(), outer2 = set()) :
  master = get_master_idx((idx1, idx2))
  assert type(master[0]) == type(master[1])
  if master[0] == master[1] and master[0] in outer1 and master[1] in outer2 :
    shifts = get_shifts((idx1, idx2))
    if shifts[0] != shifts[1] :
      return False
  else :
    if Range(idx1.lower, idx1.upper + 1).is_disjoint(Range(idx2.lower, idx2.upper + 1)) :
      return False
  return True

def block_shape(indices) :
  if indices :
    return tuple(map(len, indices))
  else :
    return (S.One,)

def block_size(indices) :
  return reduce(mul_as_func, block_shape(indices))

def block_copy(indices) :
  return tuple(t(IDummy(), (i.lower, i.upper)) for t, i in zip(get_types(indices), indices))

def is_within_block(block, indices) :
  return all(starmap(lambda b, i : b.lower <= i.lower and i.upper <= b.upper, zip(block, indices)))

def idx_subs(expr, old, new) :
  indices = (old, new)
  old_master, new_master = get_master_idx(indices)
  old_shift, new_shift = get_shifts(indices)
  return expr.subs(old_master, ShiftedIdx(new_master, new_shift - old_shift))

if __name__ == "__main__" :
  from sympy import Symbol
  from sympy2ipopt.utils.test_utils import check_value_error, renum_dummy
  from sympy2ipopt.idx_type import IdxType
  from sympy2ipopt.indexed_base_with_offset import IndexedBaseWithOffset


  t1 = IdxType('t1', (0, 10))
  t2 = IdxType('t2', (2,8))
  t3 = IdxType('t3', (-5,3))
  t4 = IdxType('t4', (-9,-4))

  i1 = t1('i1')
  i2 = t2('i2')
  i3 = t3('i3')
  i4 = t4('i4')

  j1 = t1('j1', (0, 9))
  j2 = t2('j2', (5, 7))
  j3 = t3('j3', (-1, 1))
  j4 = t4('j4', (-8, -6))
  sj1 = ShiftedIdx(j1, 1)
  j = t1('j', (2, 7))
  sj2 = ShiftedIdx(j2, -2)
  sj3 = ShiftedIdx(j3, -3)
  sj4 = ShiftedIdx(j4, 2)

  k2 = t2('k2', (6, 6))
  k3 = t3('k3', (0, 0))

  sk3 = ShiftedIdx(k3, -1)

  l1= t1('l1', (10, 10))
  l2= t1('l2', (2, 4))

  b1 = IndexedBaseWithOffset('b1')
  b2 = IndexedBaseWithOffset('b2')
  r1 = IndexedBaseWithOffset('r', (1,), 3)
  r2 = IndexedBaseWithOffset('r', (1,), 5)

  assert get_indices(b1[j1, sj2, sj3, i4]) == (j1, sj2, sj3, i4)
  assert get_indices(Symbol('a')) == tuple()

  assert get_id(b2[sj1, i2, i3, sj4]) == b2
  assert get_id(Symbol('a')) == Symbol('a')

  assert get_master_idx((i1, sj2, j3, sj4)) == (i1, j2, j3, j4)

  assert get_shifts((sj1, j2, sj3, i4)) == (S.One, S.Zero, S(-3), S.Zero)

  assert get_types((i1, sj2, sj3, j4)) == (t1, t2 ,t3, t4)    

  assert is_full_range((i1, i2, i3, i4)) == True
  assert is_full_range((i1, i2, j3, i4)) == False

  assert get_idx_pos((i1, sj2, j3, sj4), {i1, j2, sj4}) == (0, 3)

  assert can_be_eq(i1, sj1, {i1}, {j1}) == True
  assert can_be_eq(j1, sj1, {j1}, {j1}) == False
  assert can_be_eq(j1, sj1, {}, {j1}) == True
  assert can_be_eq(j1, sj1, {j1}, {}) == True
  assert can_be_eq(sj3, j3, {j3}, {}) == False
  assert can_be_eq(sj3, j3, {}, {j3}) == False
  assert can_be_eq(j1, l1, {j1}, {l1}) == False
  assert can_be_eq(j1, j1, {j1}, {j1}) == True

  assert block_shape((t1, t2, t3, t4)) == (11, 7, 9, 6)
  assert block_shape((j1, sj2, i3, sj4)) == (10, 3, 9, 3)

  assert block_size((t1, t2, t3, t4)) == 4158
  assert block_size((j1, sj2, i3, sj4)) == 810

  assert renum_dummy(block_copy((i1, i2, j3))) == (t1('_Dummy_1', (0, 10)), t2('_Dummy_2', (2, 8)), t3('_Dummy_3', (-1, 1)))
  assert renum_dummy(block_copy((i1, sj2))) == (t1('_Dummy_1', (0, 10)), t2('_Dummy_2', (3, 5)))

  assert is_within_block((i1, i2, i3), (j1, j2, j3)) == True
  assert is_within_block((j1, j2, j3), (i1, i2, i3)) == False

  assert idx_subs(sj1 + j1, j1, sj2) == ShiftedIdx(j2, -1) + sj2
  assert idx_subs(sj1 + j1, sj1, sj2) == sj2 + ShiftedIdx(j2, -3)

  print('ALL TESTS HAVE BEEN PASSED!!!')
