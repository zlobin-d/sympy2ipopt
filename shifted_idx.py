# vim: set fileencoding=utf8 :
from sympy import AtomicExpr, Expr, S, sympify
from sympy2ipopt.idx_type import IdxOutOfRangeError

class ShiftedIdx(AtomicExpr) :
  '''Выражение, являющееся суммой индекса и числа, с которым можно работать как с индексом.

  :param idx: индекс.

  :param shift: сдвиг.

  Экземпляр этого типа является выражением AtomicExpr с двумя аргументами:

  #. Индекс (с типом, созданным при помощи :class:`IdxType`).
  #. Сдвиг.

  При помощи сдвига можно создавать связанные индексы,
  например :math:`i` и :math:`i-1` для разностной схемы.

  У экземпляра этого типа есть свойства ``lower`` и ``upper``,
  как у обычного индекса `sympy.Idx <https://docs.sympy.org/latest/modules/tensor/indexed.html#sympy.tensor.indexed.Idx>`_.
  Они соответствуют границам диапазона индекса с учетом сдвига ``shift``.

  Экземпляры этого типа имеют длину: функция ``len()`` для них возвращает
  количество элементов (целых чисел) в диапазоне [``self.lower``, ``self.upper``] включая границы.

  Интервал [``self.lower``, ``self.upper``] должен содержаться в диапазоне [``start``, ``end``] типа индекса ``idx``.

  Поддерживается замена индекса ``self.idx`` на экземпляр :class:`ShiftedIdx`:
  его сдвиг будет прибавлен к ``self.shift``, а индекс ``self.idx`` заменится на новый.

  >>> t1 = IdxType('t1', (0, 5))
  >>> i = t1('i', (2, 5))
  >>> si = ShiftedIdx(i, -1)
  >>> si.lower == 1 and si.upper == 4 and len(si) == 4
      True
  >>> si.subs(i, si) == ShiftedIdx(i, -2)

  .. seealso::
  
    :class:`IdxType`
  '''

  def __new__(cls, idx, shift, **kw_args) :
    idx = sympify(idx)
    shift = sympify(shift)
    options = kw_args
    if kw_args.pop('evaluate', True) and shift == S.Zero :
      return idx
    idx_type = type(idx)
    if idx.lower + shift < idx_type.start or idx.upper + shift > idx_type.end :
      raise IdxOutOfRangeError(f'Index out of range for type "{idx_type.__name__}"')
    obj = super(cls, cls).__new__(cls, idx, shift, **kw_args)
    obj.__options = options
    return obj

  @property
  def idx(self) :
    '''Возвращает индекс.'''
    return self.args[0]

  @property
  def shift(self) :
    '''Возвращает сдвиг.'''
    return self.args[1]

  @property
  def lower(self) :
    '''Возвращает левую границу диапазона: ``self.idx.lower + self.shift``.'''
    return self.idx.lower + self.shift

  @property
  def upper(self) :
    '''Возвращает правую границу диапазона: ``self.idx.upper + self.shift``.'''
    return self.idx.upper + self.shift

  @property
  def free_symbols(self) :
    '''Сам экземпляр :class:`ShiftedIdx` также является свободным символом.'''
    return {self} | self.idx.free_symbols | self.shift.free_symbols

  @property
  def expr_free_symbols(self) :
    '''Для :class:`ShiftedIdx` совпадает с self.free_symbols.'''
    return self.free_symbols

  def __len__(self) :
    '''Количество элементов в диапазоне.'''
    return self.idx.__len__()

  def _to_expr(self) :
    '''Возвращает нижнюю границу ``self.lower``, если диапазон состоит из одного числа, иначе --- сумму ``Add(self.idx, self.shift)``.'''
    return self.lower if self.upper == self.lower else self.idx + self.shift

  def _sympystr(self, p):
    '''При распечатывании будет "(``self.idx`` + ``self.shift``)".'''
    return '(' + p.doprint(self.idx + self.shift) + ')'

  def xreplace(self, *args, **kwargs) :
    return super(Expr, self).xreplace(*args, **kwargs)

  def _eval_subs(self, old, new) :
    '''Отдельно обрабатываем замену индекса ``self.idx`` на экземпляр :class:`ShiftedIdx`: индекс заменяется, сдвиги суммируются.'''
    if old == self.idx :
      if isinstance(new, type(self)) :
        return type(self)(new.idx, new.shift + self.shift, **new.__options)
    return None


if __name__ == "__main__" :
  from sympy2ipopt.idx_type import IdxType
  from sympy2ipopt.utils.test_utils import check_value_error

  t1 = IdxType('t1', (0, 10))
  t2 = IdxType('t2', (2,8))
  t3 = IdxType('t3', (-5,3))
  t4 = IdxType('t4', (-9,-4))

  j1 = t1('j1', (0, 9))
  j2 = t2('j2', (5, 7))
  j3 = t3('j3', (-1, 1))
  j4 = t4('j4', (-8, -6))
  assert ShiftedIdx(j1, 0) == j1
  assert ShiftedIdx(j1, 0, evaluate = False).shift == S.Zero
  assert ShiftedIdx(j1, 0, evaluate = False).idx == j1
  check_value_error(ShiftedIdx, j1, 2)
  check_value_error(ShiftedIdx, j1, -1)
  sj1 = ShiftedIdx(j1, 1)
  assert sj1.free_symbols == {sj1, sj1.idx} and sj1.expr_free_symbols == {sj1, sj1.idx}
  assert sj1.idx == j1 and sj1.shift == S.One and sj1.lower == j1.lower + 1 and sj1.upper == j1.upper + 1
  assert len(sj1) == 10
  assert sj1._to_expr() == j1 + 1
  assert str(sj1) == '(j1 + 1)'
  j = t1('j', (2, 7))
  assert sj1.subs(j1, ShiftedIdx(j, -1)) == j
  assert sj1.subs(j1, j) == ShiftedIdx(j, 1)
  assert sj1.subs(j2, j) == sj1
  assert sj1.subs(j1, ShiftedIdx(j, -2)) == ShiftedIdx(j, -1)
  sj2 = ShiftedIdx(j2, -2)
  sj3 = ShiftedIdx(j3, -3)
  sj4 = ShiftedIdx(j4, 2)

  k3 = t3('k3', (0, 0))

  sk3 = ShiftedIdx(k3, -1)
  assert sk3._to_expr() == S(-1)

  print('ALL TESTS HAVE BEEN PASSED!!!')
