#!/bin/python3

from collections import defaultdict 
from sympy import Idx, Indexed, Sum, Product, Symbol
from sympy.core.function import AppliedUndef
from sympy2ipopt.utils.idx_utils import get_master_idx, get_shifts, get_types, can_be_eq, is_within_block, get_idx_pos
from sympy2ipopt.shifted_idx import ShiftedIdx
from sympy2ipopt.utils.expr_utils import get_outer_indices
from sympy.concrete.expr_with_intlimits import ExprWithIntLimits

def check_idx(expr) :
  '''Проверяем, нет ли различных индексов с одинаковым строковым представлением,
     все ли типы индексов имеют атрибуты диапазона "start" и "end".'''

  # Проверяем не только free_symbols, но и индексы суммирования и т. п.
  indices = expr.atoms(Idx)
  used = {}
  for idx in indices :
    t = type(idx)
    if not (hasattr(t, 'start') and hasattr(t, 'end')) :
      print('Type of index should have "start" and "end" attributes. See "IdxType" class.')
      return False
    if used.setdefault(str(idx), idx) != idx :
      print(f'Unequal indices with same str() representation "{idx}"')
      return False
  return True

def check_symbols(expr, user_data = {}, variables = {}) :
  '''Проверяем Indexed и Symbols в выражении: что это либо переменная, либо данные.
     Для Indexed: соответствуют ли количество индексов и типы индексов тому, что указано при регистрации.'''

  free_symbols = expr.free_symbols
  indexed = set(sym for sym in free_symbols if isinstance(sym, Indexed))
  free_symbols -= indexed
  known_symbols = set()
  for sym in indexed :
    _, indices = variables.get(sym.base, (None, None))
    # Проверку на принадлежность sym и к переменным, и к данным имеет смысл делать только после запрета добавления переменных и данных
    if indices == None :
      indices = user_data.get(sym.base, None)
    if indices != None :
      idx_types = get_types(indices)
      if len(sym.indices) != len(indices) :
        print(f'Wrong number of indices for "{sym}"')
        return False
      if any(map(lambda idx, t : not isinstance(idx, t), get_master_idx(sym.indices), idx_types)) :
        print(f'Wrong index types for "{sym}"')
        return False
    else :
      print(f'Unknown IndexedBase instance "{sym.base}"')
      return False
    known_symbols.add(sym.base.label)
  for sym in free_symbols - known_symbols :
    if isinstance(sym, Symbol) :
      in_variables = sym in variables
      in_user_data = sym in user_data
      # Проверку на (in_variables and in_user_data) имеет смысл делать только после запрета добавления переменных и данных
      if not in_variables and not in_user_data :
        print(f'Unknown Symbol instance "{sym}"')
        return False
    elif not isinstance(sym, (Idx, ShiftedIdx)) :
      print(f'Unknown free symbol "{sym}" of type "{type(sym)}"')
      return False
  return True

def check_functions(expr, user_functions = {}) :
  '''Проверяем Function в выражении: что это пользовательская функция или стандартная функция.'''

  for sym in expr.atoms(AppliedUndef) :
    if sym.func not in user_functions :
      print(f'Unknown AppliedUndef instance "{sym.func}"')
      return False
  return True

def check_for_disjoint_block_vars(expr, variables) :
  '''Проверяем, что выражение обладает свойством постоянства структуры.
  
  Предполагаем, что все внутренние индексы -- уникальные символы.
  Это нужно, чтобы избежать совпадения имен внутренних индексов между собой или с внешними индексами.
  '''

  outer_indices = get_outer_indices(expr)
  bases = defaultdict(dict)
  def check (expr) :
    if isinstance(expr, Indexed) :
      if expr.base in variables :
        outer_pos = get_idx_pos(get_master_idx(expr.indices), outer_indices)
        outer = tuple(expr.indices[n] for n in outer_pos)
        for disjoint_indices, disjoint_outer in bases[expr.base].items() :
          if outer != disjoint_outer :
            if all(map(lambda i1, i2 : can_be_eq(i1, i2, get_master_idx(outer), get_master_idx(disjoint_outer)), expr.indices, disjoint_indices)) :
              print(f'Non-constant structure of expr due to indexed variable "{expr.base}".')
              return False
        bases[expr.base][expr.indices] = outer
      return True
    for arg in expr.args :
      if not check(arg) :
        return False
    return True
  return check(expr)

def check_for_exprs_with_int_limits(expr) :
  '''Проверяем, что из ExprWithIntLimits есть только Sum.'''

  for e in expr.atoms(ExprWithIntLimits) :
    if not isinstance(e, Sum) :
      print(f'Unsupported expr with limits "{type(e).__name__}".')
      return False
  return True

if __name__ == "__main__" :
  from sympy import IndexedBase, cos, Atom, Function
  from sympy2ipopt.idx_type import IdxType
  from sympy2ipopt.utils.idx_utils import IDummy
  r = IndexedBase('r')
  p = IndexedBase('p')
  f = IndexedBase('f')
  g = IndexedBase('g')

  t1 = IdxType('t1', (1, 3))
  t2 = IdxType('t2', (2, 5))

  i1 = t1('i1')
  i2 = t2('i2')

  j1 = t1('j1', (2, 3))
  j2 = t2('j2', (2, 4))

  sj1 = ShiftedIdx(j1, -1)
  sj2 = ShiftedIdx(j2, 1)

  variables = {r : (0, (i1, i2)), p : (12, (i1, i2))}

  assert check_idx(r[t1('i', (1, 2))]) == True
  assert check_idx(r[Idx('i', (1, 2))]) == False
  assert check_idx(r[t1('i', (1, 2))] + p[t1('i', (1, 2))]) == True
  assert check_idx(r[t1('i', (1, 2))] + p[t1('i', (1, 3))]) == False

  assert check_symbols(r[i1], {}, variables) == False
  assert check_symbols(r[i1], variables, {}) == False
  assert check_symbols(r[i2, i1] + p[sj1, j2]**2, {}, variables) == False
  assert check_symbols(r[i2, i1] + p[sj1, j2]**2, variables, {}) == False
  assert check_symbols(r[i1, i1], {}, variables) == False
  assert check_symbols(r[i1, i1], variables, {}) == False
  assert check_symbols(IndexedBase(Symbol('z'))[i2, i1, j2] + p[i1, sj2]**2, {}, variables) == False
  assert check_symbols(IndexedBase(Symbol('z'))[i2, i1, j2] + p[i1, sj2]**2, variables, {}) == False
  z = IndexedBase(Symbol('z'), shape = (4, 3, 4))
  assert check_symbols(z[i2, i1, j2] + p[i1, sj2]**2, {z : (i2, i1, j2)}, variables) == True
  x = Symbol('x')
  assert check_symbols(x**2, {z : (i2, i1, j2)}, variables) == False
  assert check_symbols(x**2, {x : ()}, variables) == True
  assert check_symbols(x**2, variables, {x : ()}) == True
  class Test(Atom) :
    @property
    def free_symbols(self) :
      return {self}
  assert check_symbols(Test(), {}, variables) == False

  G = Function('G')
  assert check_functions(G() + 5, {}) == False
  assert check_functions(G() + 5, {G : None}) == True

  assert check_for_disjoint_block_vars(r[i1, i2] + r[j1, i2]**2, variables) == False
  assert check_for_disjoint_block_vars(r[sj1, j2] + r[sj1, sj2]**2, variables) == True
  h1 = t1('h1', (1, 1))
  h2 = t2('h2', (5, 5))
  cj1 = j1.subs(j1.label, IDummy())
  csj1 = ShiftedIdx(cj1, -1)
  ci2 = i2.subs(i2.label, IDummy())
  assert check_for_disjoint_block_vars(r[j1, j2] + r[h1, h2]**2, variables) == True
  assert check_for_disjoint_block_vars(Sum(r[cj1, j2], cj1) + r[j1, j2]**2, variables) == False
  assert check_for_disjoint_block_vars(Sum(r[cj1, j2] + r[csj1, j2], cj1) + r[h1, j2]**2, variables) == True
  assert check_for_disjoint_block_vars(Sum(r[cj1, j2] + r[csj1, j2], cj1) + r[i1, j2]**2, variables) == False
  assert check_for_disjoint_block_vars(Sum(r[cj1, i2], cj1) + r[j1, j2]**2, variables) == False
  assert check_for_disjoint_block_vars(Sum(r[j1, ci2], ci2) + r[j1, j2]**2, variables) == False

  sum1 = Sum(r[i1, j2], i1, j2)
  sum2 = Sum(p[sj1, sj2], j2)
  sum3 = Sum(f[i1] + sum1, i1)
  prod1 = Product(r[i1, j2], i1, j2)
  i3 = IdxType('t3', (-5,3))('i3')
  i4 = IdxType('t4', (-9,-4))('i4')
  sum4 = Sum(r[i1, j2] + g[i3, i4], i1, i3, i4)
  assert check_for_exprs_with_int_limits(sum3**8) == True
  assert check_for_exprs_with_int_limits(5 + prod1) == False
  assert check_for_exprs_with_int_limits(cos(sum2)) == True
  assert check_for_exprs_with_int_limits(sum1) == True
  assert check_for_exprs_with_int_limits(sum4) == True

  print('ALL TESTS HAVE BEEN PASSED!!!')

