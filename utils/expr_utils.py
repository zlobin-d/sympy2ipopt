#!/bin/python3

from collections import defaultdict
from sympy import Idx, IndexedBase, Indexed, Sum, Dummy, UnevaluatedExpr, Tuple
from sympy2ipopt.utils.idx_utils import get_master_idx, get_idx_pos, IDummy

def RDummy() :
  return Dummy(real = True)

def to_expr(expr) :
  if isinstance(expr, UnevaluatedExpr) :
    return expr
  if hasattr(expr, '_to_expr') :
    expr = expr._to_expr()
  return expr.func(*(to_expr(arg) for arg in expr.args)) if expr.args else expr

def get_sums(expr) :
  '''Находим все вхождения Sum первого уровня вложенности.'''
  sums = []
  def get(expr) :
    if isinstance(expr, Sum) :
      sums.append(expr)
    else :
      for arg in expr.args :
        get(arg)
  get(expr)
  return sums

def get_outer_indices(expr) :
  '''Находим все внешние индексы выражения.'''
  return tuple(sorted((sym for sym in expr.free_symbols if isinstance(sym, Idx) and len(sym) > 1), key = lambda idx : str(idx)))

def wrap_in_sum(term, indices) :
  '''Сумма term по индексам indices. 

  Если диапазон индекса состоит из одного значения, то не суммируем по этому индексу.
  Если индекс не является свободным индексом для term, то просто домножаем term на длину диапазона индекса.
  '''
  #free_symbols = term.free_symbols
  outer_indices = get_outer_indices(term)
  sum_indices = []
  for idx in reversed(get_master_idx(indices)) :
    if idx.lower == idx.upper :
      continue
    if idx in outer_indices :
      sum_indices.append(idx)
    else :
      # Если в term нет индекса, то просто умножаем на количество итераций
      term *= len(idx)
  return Sum(term, *sum_indices) if sum_indices else term

def prepare_expr(expr) :
  '''Делаем все индексы суммирования уникальными.

  Нужно для упрощения их последующей обработки:
  чтобы не оказалось двух разных внутренних или внешнего и внутреннего индекса с одинковыми именами.
  '''

  # здесь нужны внешние индексы, а также свободные индексы длины 1
  all_indices = {sym for sym in expr.free_symbols if isinstance(sym, Idx)}
  def prepare(expr) :
    if isinstance(expr, Sum) :
      term, *indices = expr.args
      term = prepare(term)
      new_indices = tuple((val[0].subs(val[0].label, IDummy()), val[1], val[2]) if val[0] in all_indices else val for val in indices)
      term = term.subs({old[0] : new[0] for old, new in zip(indices, new_indices)})
      all_indices.update({idx for idx, *_ in indices})
      return expr.func(term, *new_indices)
    return expr.func(*(prepare(arg) for arg in expr.args)) if expr.args else expr
  return prepare(expr)

# Чтобы to_expr уже было определено. По хорошему нужно перепроектировать to_disjoint_parts, чтобы не было круговой зависимости.
from sympy2ipopt.utils.block_utils import Part, to_disjoint_parts
def expr_var_indices(expr, var_id, additional_indices = list()) :
  '''Все переменные блока ``var_id`` от которых зависит ``expr``.
  
  ``additional_indices`` --- список дополнительных наборов индексов.
  Тех, которых нет в ``expr``, но которые нужно рассмотреть наряду с используемыми в выражении.
  Например набор, описывающий полный блок ``var_id``. Индексы этих наборов не учитываются при определении внешних индексов ``expr``.
  Возвращает множество пар (набор индексов; набор вхождений, в индексы которых вложен этот набор индексов). Для простой переменной {((), ())}.

  Предполагаем, что все внутренние индексы -- уникальные символы.
  Это нужно, чтобы избежать совпадения имен внутренних индексов между собой или с внешними индексами.
  '''

  if not isinstance(var_id, IndexedBase) :
    return {((), ())}
  outer_indices = get_outer_indices(expr)
  # словарь:
  # ключ --- внешние индексы набора,
  # значение --- множество вхождений в выражение объектов Indexed с такими же внешними индексами, как в ключе.
  outer_map = defaultdict(set)
  def add_to_outer_map(indexed) :
    outer_pos = get_idx_pos(get_master_idx(indexed.indices), outer_indices)
    outer = frozenset((n, indexed.indices[n]) for n in outer_pos)
    outer_map[outer].add(indexed)
  def get(expr) :
    if isinstance(expr, Indexed) and expr.base == var_id :
      add_to_outer_map(expr)
    else :
      for arg in expr.args :
        get(arg)
  get(expr)
  for indices in additional_indices :
    add_to_outer_map(var_id[indices])
  # Для применения to_disjoint_parts. По хорошему нужно перепроектировать to_disjoint_parts.
  class  ItemsBox(tuple) :
    def subs(self, *args, **kwargs) :
      return self
    def __add__(self, other) :
      return type(self)((*self, *other))
    def __iadd__(self, other) :
      return type(self)((*self, *other))
  # множество пар: (набор индексов; набор вхождений, в индексы которого вложен этот набор индексов)
  indices_set = set()
  for outer, elems in outer_map.items() :
    parts = []
    for e in sorted(elems, key = str) :
      indices = tuple(val[1] for val in enumerate(e.indices) if val not in outer)
      parts.append(Part(outer, indices, ItemsBox((e,))))
    parts = to_disjoint_parts(parts)
    for p in parts :
      indices = [None] * (len(p.indices) + len(p.block_id))
      for n, idx in p.block_id :
        indices[n] = idx
      for n, idx in zip(get_idx_pos(indices, {None}), p.indices) :
        indices[n] = idx
      indices_set.add((tuple(indices), p.term))
  return indices_set


if __name__ == "__main__" :
  from sympy2ipopt.idx_type import IdxType
  from sympy2ipopt.shifted_idx import ShiftedIdx
  from sympy2ipopt.indexed_base_with_offset import IndexedBaseWithOffset
  from sympy2ipopt.utils.test_utils import renum_dummy
  from sympy import Symbol,srepr

  dummy = RDummy()

  b1 = IndexedBaseWithOffset('b1')
  b2 = IndexedBaseWithOffset('b2')
  r1 = IndexedBaseWithOffset('r', (1,), 3)
  r2 = IndexedBaseWithOffset('r', (1,), 5)

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
  sj2 = ShiftedIdx(j2, -2)
  sj3 = ShiftedIdx(j3, -3)
  sj4 = ShiftedIdx(j4, 2)

  k3 = t3('k3', (0, 0))
  sk3 = ShiftedIdx(k3, -1)

  assert to_expr(b1[j1, UnevaluatedExpr(sj2), sj3, i4] + 5 * (r1[i1] + r2[i1])) == b1[j1, UnevaluatedExpr(sj2), j3 - 3, i4] + 5 * r1[i1] + 5 * r2[i1]

  r1 = IndexedBaseWithOffset('r', (1,), 3)

  l1= t1('l1', (10, 10))

  sum0 = Sum(b1[i1, k3], i1, k3)
  sum1 = Sum(b2[sj2, sj3], j2)
  sum2 = Sum(b1[i1, j2] + b2[i3, i4], i1, i3, i4)
  sum3 = Sum(r1[i1] + sum0, i1)
  assert get_sums(sum0) == [sum0]
  assert get_sums(sum3) == [sum3]
  assert get_sums(5 * sum3 * sum1**3) == [sum1, sum3]

  assert wrap_in_sum(b2[l1, sj3, i3], (i2, l1, sj3, sj4, i3, sk3)) == Sum(21 * b2[l1, sj3, i3], i3, j3)

  assert get_outer_indices(b1[i1] + r1[sj3]**b2[j2]) == (i1, j2, j3)
  assert get_outer_indices(b1[i1] + sum1) == (i1, j3)
  assert get_outer_indices(b1[i1] + sum2) == (i1, j2)
  assert get_outer_indices(b1[i1]*b2[sj1, sk3, i4]) == (i1, i4, j1)

  x = Symbol('x')
  assert renum_dummy(prepare_expr(x*Sum(x * k3 / r1[i1] + Sum(5 * b1[i1, k3], i1, k3), i1))) == x * Sum(x * k3 / r1[t1('_Dummy_2', (0, 10))] + Sum(5 * b1[i1, t3('_Dummy_1', (0, 0))], i1, t3('_Dummy_1', (0, 0))), t1('_Dummy_2', (0, 10)))

  expr = x**2/b1[i1,j2]
  assert prepare_expr(expr) == expr
  assert renum_dummy(prepare_expr(sum0 + k3)) == Sum(b1[i1, t3('_Dummy_1', (0, 0))], i1, t3('_Dummy_1', (0, 0))) + k3

  assert expr_var_indices(b2[l1, sj3] + b1[l1, sj3]**b1[j1, sj3] + b1[l1, j3]*b1[j1, j3], b1) == {((l1, sj3), (b1[l1, sj3],)), ((j1, sj3), (b1[j1, sj3],)), ((j1, j3), (b1[j1, j3],)), ((l1, j3), (b1[l1, j3],))}
  assert expr_var_indices(b2[i1, sj2] + b1[i1, sj2], b1) == {((i1, sj2), (b1[i1, sj2],))}
  assert expr_var_indices(Symbol('y')*2 + Symbol('z')**b2[i1, sj2], Symbol('z')) == {((), ())}
  assert renum_dummy(expr_var_indices(Sum(b1[i1, j2] + b1[sj1, j2], j1, i1), b1)) == {((t1('_Dummy_1', (1, 10)), j2), (b1[sj1, j2], b1[i1, j2])), ((t1('_Dummy_2', (0, 0)), j2), (b1[i1, j2],))}
  assert renum_dummy(expr_var_indices(Sum(b1[i1, j2] + b1[sj1, sj2], j1, i1, j2), b1)) == {((t1('_Dummy_1', (1, 10)), t2('_Dummy_5', (6, 7))), (b1[i1, j2],)), ((t1('_Dummy_1', (1, 10)), t2('_Dummy_4', (3, 4))), (b1[sj1, sj2],)), ((t1('_Dummy_2', (0, 0)), j2), (b1[i1, j2],)), ((t1('_Dummy_1', (1, 10)), t2('_Dummy_3', (5, 5))), (b1[sj1, sj2], b1[i1, j2]))}
  assert expr_var_indices(b2[i1, sj2] + b1[i1, sj2], b1, [(j1, j2)]) == {((i1, sj2), (b1[i1, sj2],)), ((j1, j2), (b1[j1, j2],)) }
  assert renum_dummy(expr_var_indices(Sum(b1[i1, sj2], i1), b1, [(j1, sj2)])) == {((t1('_Dummy_1', (0, 9)), sj2), (b1[i1, sj2], b1[j1, sj2])), ((t1('_Dummy_2', (10, 10)), sj2), (b1[i1, sj2],))}

  print('ALL TESTS HAVE BEEN PASSED!!!')

