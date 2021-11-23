# vim: set fileencoding=utf8 :

from sympy.core.assumptions import ManagedProperties
from sympy import Idx, sympify
from builtins import range as builtin_range

class IdxOutOfRangeError(ValueError) :
  '''Исключение выхода за границу типа индекса.'''
  pass

class IdxType(ManagedProperties) :
  '''Метакласс, экземпляры которого являются типами для индексов.

  Экземпляры :class:`IdxType` являются типами,
  производными от `sympy.Idx <https://docs.sympy.org/latest/modules/tensor/indexed.html#sympy.tensor.indexed.Idx>`_.

  :param name: имя создаваемого типа.

  :param limits: диапазон типа: набор из двух целочисленных элементов.
    Второй элемент должен быть больше или равен первого (обратные индексы не поддерживаются).
    Диапазоны индексов-экземпляров создаваемого типа обязательно будут содержаться в диапазоне ``limits``.
    Диапазон состоит из целых чисел, включая границы.

  Индексы-экземпляры создаваемого типа отличаются от экземпляров `sympy.Idx <https://docs.sympy.org/latest/modules/tensor/indexed.html#sympy.tensor.indexed.Idx>`_:

  #. Имеют длину: функция ``len()`` для них возвращает количество элементов в диапазоне [``lower``, ``upper``] включая границы.
  #. Имеют метод ``_to_expr()``, который возвращает нижнюю границу ``lower``,
     если диапазон состоит из одного числа, а в противном случае --- сам индекс без изменения.

  Если в конструктор создаваемого типа передать два аргумента,
  то его поведение будет аналогично`sympy.Idx <https://docs.sympy.org/latest/modules/tensor/indexed.html#sympy.tensor.indexed.Idx>`_.

  Дополнительно можно передавать один аргумент ``arg``:
     
  * Если он целочисленный, то будет создан индекс с именем "``name``\\(``arg``)" и диапазоном из одного элемента.
  * Иначе будет создан индекс с именем "``arg``" и диапазоном совпадающим с диапазоном типа ``limits``.


  >>> t1 = IdxType('t1', (0, 5))
  >>> t1.start == 0 and t1.end == 5
      True
  >>> for val in t1.range() :
  ...   print(val)
  ...
      0
      1
      2
      3
      4
      5
  >>> i = t1('i')
  >>> i.lower == 0 and i.upper == 5 and i.label = Symbol('i', integer = True)
      True
  >>> j = t1(2)
  >>> j.lower == 2 and j.upper = 2 and j.label = Symbol('t1(2)', integer = True)
      True
  >>> k = t1('k', (1, 4))
  >>> k.lower == 1 and k.upper == 4 and k.label = Symbol('k', integer = True)
      True
  >>> len(t1) == 6 and len(i) == 6 and len(j) == 1 and len(k) == 4
      True
  >>> i._to_expr() == i and j._to_expr() == 2
      True
  '''

  def __new__(cls, name, limits) :
    limits = sympify(limits)
    if limits[0] > limits[1] :
      print(f'Incorrect limits for {cls.__name__}')
      raise ValueError
    def t_new(cls, *args, **kw_args) :
      if len(args) == 1 :
        if isinstance(args[0], int) or getattr(args[0], 'is_Integer', False) :
          args = (f'{cls.__name__}({args[0]})', (args[0], args[0]))
        else :
          args += ((limits[0], limits[1]),)
      obj = super(cls, cls).__new__(cls, *args, **kw_args)
      if obj.lower > obj.upper :
        print(f'Reverse indexing is not supported for type "{cls.__name__}"')
        raise ValueError
      if obj.lower < limits[0] or obj.upper > limits[1] :
        raise IdxOutOfRangeError(f'Index out of range for type "{cls.__name__}"')
      return obj
    def t_len(self) :
      return self.upper - self.lower + 1
    def t_to_expr(self) :
      return self.lower if self.upper == self.lower else self
    obj = super().__new__(cls, name, (Idx,), {'__new__' : t_new, '__len__' : t_len, '_to_expr' : t_to_expr})
    obj.__limits = limits
    return obj

  def __len__(self) :
    '''Количество элементов в диапазоне типа.'''

    return self.end - self.start + 1

  @property
  def start(self) :
    '''Возвращает левую границу диапазона типа.'''

    return self.__limits[0]

  @property
  def end(self) :
    '''Возвращает правую границу диапазона типа.'''

    return self.__limits[1]

  def range(self) :
    '''Возвращает последовательность элементов диапазона типа --- целые числа.'''

    return builtin_range(self.start, self.end + 1)

if __name__ == "__main__" :
  from sympy import S
  from sympy2ipopt.utils.test_utils import check_value_error

  check_value_error(IdxType, 't', (3, 0))

  t1 = IdxType('t1', (0, 10))
  assert len(t1) == 11 and t1.start == S.Zero and t1.end == S(10)
  assert t1(0).lower == S.Zero and t1(0).upper == S.Zero and str(t1(0)) == 't1(0)'
  assert list(t1.range()) == list(range(11))
  t2 = IdxType('t2', (2,8))
  t3 = IdxType('t3', (-5,3))
  t4 = IdxType('t4', (-9,-4))
  assert t4(-5).lower == -5 and t4(-5).upper == -5 and str(t4(-5)) == 't4(-5)'

  i1 = t1('i1')
  assert len(i1) == 11 and i1.lower == S.Zero and i1.upper == S(10)
  i2 = t2('i2')
  i3 = t3('i3')
  i4 = t4('i4')

  check_value_error(t1, 'i', (3, 0))
  check_value_error(t1, 'i', (3, 20))
  check_value_error(t1, 'i', (-1, 3))
  j1 = t1('j1', (0, 9))
  assert len(j1) == 10 and j1.lower == S.Zero and j1.upper == S(9)
  assert j1._to_expr() == j1

  k2 = t2('k2', (6, 6))
  assert k2._to_expr() == 6
  k3 = t3('k3', (0, 0))

  print('ALL TESTS HAVE BEEN PASSED!!!')
