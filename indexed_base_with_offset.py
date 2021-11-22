from sympy import IndexedBase, S, sympify

class IndexedBaseWithOffset(IndexedBase) :
  ''' IndexedBase, в котором offset является аргументом выражения.
      Нужно, чтобы различать переменные IpOpt по offset. Без этого :
      >>> x1 = IndexedBase('x', offset = 3)
      >>> x2 = IndexedBase('x', offset = 5)
      >>> i = Idx('i', (0,5))
      >>> x1 == x2
      True
      >>> x1 + x2 == 2 * x1
      True
      >>> str(x1[i])
      'x'
      >>> str(x2[i])
      'x'
      >>> ccode(x1[i])
      'x[i + 3]'
      >>> ccode(x2[i])
      'x[i + 5]'
      >>> ccode(x1[i] + x2[i])
      '2*x[i + 5]'
      '''
  def __new__(cls, label, *args, strides = None, **kw_args) :
    n_args = len(args)
    if n_args > 2 :
      print('IndexedBaseWithOffset: too many args')
      raise ValueError
    is_offset = 'offset' in kw_args
    is_shape = 'shape' in kw_args
    if n_args == 0 :
      shape = kw_args.pop('shape', None)
      offset = kw_args.pop('offset', S.Zero)
    elif n_args == 1 :
      if is_offset and is_shape :
        print('IndexedBaseWithOffset: multiple values for offset or shape.')
        raise ValueError
      if is_offset :
        shape = args[0]
        offset = kw_args.pop('offset')
      else :
        shape = kw_args.pop('shape', None)
        offset = args[0]
    else :
      if is_offset or is_shape :
        print('IndexedBaseWithOffset: multiple values for offset and/or shape.')
        raise ValueError
      shape, offset = args
    offset = sympify(offset)
    obj = super().__new__(cls, label, shape, offset = offset, strides = strides, **kw_args)
    obj._args += (offset,)
    return obj

  # Для отладки
  #def _sympystr(self, p) :
  #  return f'({p.doprint(self.label + self.offset)})'

  @property
  def offset(self) :
    return self.args[-1]

if __name__ == "__main__" :
  from sympy import Tuple, ccode
  from sympy2ipopt.utils.test_utils import check_value_error
  from sympy2ipopt.idx_type import IdxType

  r1 = IndexedBaseWithOffset('r', (1,), 3)
  r2 = IndexedBaseWithOffset('r', (1,), 5)

  r = IndexedBaseWithOffset('r', 5)
  assert r.args == (r.label, S(5))

  r = IndexedBaseWithOffset('r', offset = 5, shape = (1,))
  assert r.args == (r.label, Tuple(S.One), S(5))
  r = IndexedBaseWithOffset('r', offset = 5)
  assert r.args == (r.label, S(5))
  r = IndexedBaseWithOffset('r', shape = (1,))
  assert r.args == (r.label, Tuple(S.One), S.Zero)

  r = IndexedBaseWithOffset('r', (1,), offset = 5)
  assert r.args == (r.label, Tuple(S.One), S(5))
  r = IndexedBaseWithOffset('r', 5, shape = (1,))
  assert r.args == (r.label, Tuple(S.One), S(5))
  check_value_error(IndexedBaseWithOffset, 'r', 5, shape = (1,), offset = 5)
  check_value_error(IndexedBaseWithOffset, 'r', (1,), 5, shape = (1,))
  check_value_error(IndexedBaseWithOffset, 'r', (1,), 5, offset = 5)
  check_value_error(IndexedBaseWithOffset, 'r', (1,), 5, shape = (1,), offset = 5)

  t1 = IdxType('t1', (0, 10))
  i1 = t1('i1')
  #assert str(r1) == '(r + 3)'
  #assert str(r2) == '(r + 5)'
  assert r1.offset == 3
  assert r2.offset == 5
  assert r1 != r2
  assert r1 + r2 != 2 * r1 and r1 + r2 != 2 * r2
  assert ccode(r1[i1]) == 'r[i1 + 3]'
  assert ccode(r2[i1]) == 'r[i1 + 5]'
  assert ccode(r1[i1] + r2[i1], order = 'none') == 'r[3 + i1] + r[5 + i1]'

  print('ALL TESTS HAVE BEEN PASSED!!!')
