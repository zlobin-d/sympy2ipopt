#!/bin/python3

from sympy import Symbol, Dummy, Basic
from itertools import starmap
from re import findall, search

def check_value_error(func, *args, **kw_args) :
  try :
    func(*args, **kw_args)
    assert False
  except ValueError :
    pass
  except :
    assert False

def check_runtime_error(func, *args, **kw_args) :
  try :
    func(*args, **kw_args)
    assert False
  except RuntimeError :
    pass
  except :
    assert False

def check_type_error(func, *args, **kw_args) :
  try :
    func(*args, **kw_args)
    assert False
  except TypeError :
    pass
  except :
    assert False

def check_limits(indices, limits) :
  return all(starmap(lambda idx, lim : idx.lower == lim[0] and idx.upper == lim[1], zip(indices, limits)))

def renum_dummy(arg) :
  entries = set()
  def collect_dummy(arg) :
    if isinstance(arg, str) :
      entries.update(findall('_Dummy_\d+', arg))
    else :
      try :
        iterator = iter(arg)
      except TypeError :
        if isinstance(arg, Basic) :
          entries.update(arg.atoms(Dummy))
      else :
        for a in arg :
          collect_dummy(a)
  collect_dummy(arg)
  entries = sorted(entries, key = lambda e : int(search('\d+', str(e))[0]))
  if entries :
    shift = int(search('\d+', str(entries[0]))[0]) - 1
  subs = {}
  for d in entries :
    n = int(search('\d+', str(d))[0]) - shift
    subs[d] = f'_Dummy_{n}' if isinstance(d, str) else Symbol(f'_Dummy_{n}', integer = True)
  subs_str = dict(filter(lambda s : isinstance(s[0], str), subs.items()))
  subs_expr = dict(filter(lambda s : isinstance(s[0] , Symbol), subs.items()))
  def subs_dummy(arg) :
    if isinstance(arg, str) :
      for elem in subs_str.items() :
        arg = arg.replace(*elem)
      return arg
    else :
      try:
        iterator = iter(arg)
      except TypeError:
        if isinstance(arg, Basic) :
          return arg.xreplace(subs_expr)
        else :
          return arg
      else:
        new_arg = []
        for a in arg :
          new_arg.append(subs_dummy(a))
        return type(arg)(new_arg)
  return subs_dummy(arg)

if __name__ == "__main__" :
  from sympy import Idx

  def r_err() :
    raise RuntimeError
  def v_err() :
    raise ValueError
  def t_err() :
    raise TypeError

  check_runtime_error(r_err)
  check_value_error(v_err)
  check_type_error(t_err)

  assert check_limits((Idx('i', (1, 5)),), [(1, 5)]) == True
  assert check_limits((Idx('i', (1, 5)),), [(1, 6)]) == False
  assert check_limits((Idx('i', (1, 5)),), [(2, 6)]) == False
  assert check_limits((Idx('i', (1, 5)),), [(2, 5)]) == False
  assert check_limits((Idx('i', (1, 5)), Idx('j', (3, 9))), [(1, 5), (3, 9)]) == True
  assert check_limits((Idx('i', (1, 5)), Idx('j', (4, 9))), [(1, 5), (3, 9)]) == False

  assert renum_dummy('abcd') == 'abcd'
  assert renum_dummy([[[Symbol('x'), Symbol('y')], ['abc']]]) == [[[Symbol('x'), Symbol('y')], ['abc']]]
  assert renum_dummy([['func(_Dummy_5)', '_Dummy_8 + a'], ['int _Dummy_4;']]) == [['func(_Dummy_2)', '_Dummy_5 + a'], ['int _Dummy_1;']]
  assert renum_dummy((Idx(Dummy(integer = True), (1, 7)), [Idx(Dummy(integer = True), (0, 0))], [Idx(Dummy(integer = True), (8, 8))])) == (Idx('_Dummy_1', (1, 7)), [Idx('_Dummy_2', (0, 0))], [Idx('_Dummy_3', (8, 8))])
  assert renum_dummy((Idx(Dummy(integer = True), (1, 7)), [Idx(Dummy(integer = True), (0, 0)), str(Dummy(integer = True)) + ' + a'], [f'func({Dummy()})', Idx(Dummy(integer = True), (8, 8))])) == (Idx('_Dummy_1', (1, 7)), [Idx('_Dummy_2', (0, 0)), '_Dummy_3 + a'], ['func(_Dummy_4)', Idx('_Dummy_5', (8, 8))])

  print('ALL TESTS HAVE BEEN PASSED!!!')

