# vim: set fileencoding=utf8 :

from sympy import pprint, srepr
from sympy import Symbol, S, IndexedBase, Idx, Tuple, Function, Derivative, diff, Piecewise, And, Eq, Add, UnevaluatedExpr, Sum
from sympy.codegen.ast import Element, CodeBlock, Comment, Assignment, Print
from sympy.codegen.cnodes import PreIncrement
from collections import OrderedDict, defaultdict
from itertools import takewhile, combinations, starmap
from functools import reduce

from sympy2ipopt.idx_type import IdxType
from sympy2ipopt.shifted_idx import ShiftedIdx
from sympy2ipopt.indexed_base_with_offset import IndexedBaseWithOffset
from sympy2ipopt.utils.idx_utils import get_master_idx, get_shifts, get_types, get_id, get_indices, block_shape, block_size, block_copy, IDummy, is_full_range
from sympy2ipopt.utils.expr_utils import get_outer_indices, expr_var_indices, wrap_in_sum, RDummy, prepare_expr
from sympy2ipopt.utils.code_utils import FPrint, If, elem_pos, indexed_for_array_elem, decl_for_indices, wrap_in_loop, assign_one, assign_by_indices, assign_by_counter, assign, offset_in_block, empty_operator_, cxxcode
from sympy2ipopt.utils.block_utils import cmp_with_diag, Part, to_disjoint_parts
from sympy2ipopt.utils.limitation_utils import check_idx, check_symbols, check_functions, check_for_disjoint_block_vars, check_for_exprs_with_int_limits
from sympy2ipopt.utils.diff_utils import diff_indexed

class Nlp :
  '''Описание задачи нелинейного программирования. Генерация кода в соответствии с С++ API IpOpt.

  :param name: имя для

    #. C++ класса-реализации интерфейса Ipopt::TNLP.
    #. Cгенерированных файлов:
       После успешной кодогенерации в текущей директории будут получены файлы:

       #. ``name + '.hpp'``,
       #. ``name +'.cpp'``,
       #. ``name + ' _user_decls.h'`` (при необходимости).
  '''

  # Переменные C++ интерфейса IpOpt
  __n = Symbol('n', integer = True)
  __m = Symbol('m', integer = True)
  __nnz_jac_g = Symbol('nnz_jac_g', integer = True)
  __nnz_h_lag = Symbol('nnz_h_lag', integer = True)
  __index_style = Symbol('index_style', integer = True)
  __x = Symbol('x', real = True)
  __x_l = Symbol('x_l', real = True)
  __x_u = Symbol('x_u', real = True)
  __g = Symbol('g', real = True)
  __g_l = Symbol('g_l', real = True)
  __g_u = Symbol('g_u', real = True)
  __z = Symbol('z', real = True)
  __z_L = Symbol('z_L', real = True)
  __z_U = Symbol('z_U', real = True)
  __lambda = Symbol('lambda', real = True)
  __init_x = Symbol('init_x', real = True)
  __init_z = Symbol('init_z', real = True)
  __init_lambda = Symbol('init_lambda', real = True)
  __obj_value = Symbol('obj_value', real = True)
  __obj_factor = Symbol('obj_factor', real = True)
  __grad_f = Symbol('grad_f', real = True)
  __new_x = Symbol('new_x', integer = True)
  __values = Symbol('values', real = True)
  __iRow = Symbol('iRow', integer = True)
  __jCol = Symbol('jCol', integer = True)
  __new_lambda = Symbol('new_lambda', integer = True)

  __counter = Symbol('counter', integer = True)

  __reserved_symbols = {__n, __m, __nnz_jac_g, __nnz_h_lag, __x, __x_l, __x_u, __g, __g_l, __g_u, __z, __z_L, __z_U,
                        __lambda, __init_x, __init_z, __init_lambda, __obj_value, __obj_factor, __grad_f, __new_x,
                        __new_lambda, __values, __iRow, __jCol, __counter}

  @classmethod
  def reserved_names(cls) :
    '''Возвращает множество зарезервированных имен интерфейса IpOpt.

    Нельзя добавлять пользовательские данные или функции с этими именами, так как они используются в С++ программе.
    Запрет не распространяется на переменные задачи, они могут иметь любые имена (т. к. при кодогенерации заменяются на 'x[...]').
    '''

    return set(str(s) for s in cls.__reserved_symbols)

  __nlp_upper_bound_inf = S(10.0)**19
  __nlp_lower_bound_inf = -S(10.0)**19

  class __SingleVar(Symbol) :
    '''Создает объекты-дескрипторы простой переменной.'''

    def __new__(cls, name, offset) :
      obj = Symbol.__xnew__(cls, name, real = True)
      indices = (IdxType(str(IDummy()), (0, 0))(str(IDummy())),)
      obj.__x = indexed_for_array_elem('x', indices, offset, real = True)
      obj.__offset = offset
      return obj
    @property
    def func(self) :
      # Для сохранения поведения Symbol: у Symbol.func один аргумент
      return lambda name : type(self)(name, self.__offset)
    def _to_expr(self) :
      '''Преобразование в x[...] для IpOpt.'''

      return self.__x

  class __BlockVar(IndexedBase) :
    '''Создает объекты-дескрипторы блочной переменной.'''

    def __new__(cls, name, indices, offset) :
      x = indexed_for_array_elem('x', indices, offset, real = True)
      obj = super().__new__(cls, name, shape = x.base.shape, real = True)
      obj.__x = x
      obj.__offset = offset
      return obj
    @property
    def func(self) :
      # Переопределяем func, чтобы сохранить тождество sympy: x == x.func(*x.args)
      # и не делать indices и offset аргументами
      def _func(name, shape = None) :
        if shape != None and shape != self.shape :
          print(f'Attempt to change shape of {type(self)}. Not pass "shape" argument to leave it unchanged.')
          raise ValueError
        return type(self)(name, self.__x.indices, self.__offset)
      return _func
    def _to_expr(self) :
      '''Преобразование в x[...] для IpOpt.'''

      return self.__x.base

  def __init__(self, name = 'generated_nlp') :
    self.__name = name
    self.__vars_is_added = False
    self.__target_and_constrs_is_added = False
    # Порядок нужен, чтобы задать нумерацию переменных
    self.__variables = OrderedDict()
    # Порядок нужен, чтобы задать нумерацию ограничений
    self.__constraints = OrderedDict()
    self.__var_offset = S.Zero
    self.__var_decl = set()
    self.__constr_offset = S.Zero
    self.__constr_decl = set()
    self.__objective = None
    self.__get_bounds_info_body = []
    self.__get_starting_point_body = []
    self.__eval_f_body = []
    self.__eval_g_body = []
    self.__eval_grad_f_decl = set()
    self.__eval_grad_f_body = []
    self.__jac_non_zeros = S.Zero
    self.__eval_jac_g_decl = set()
    self.__eval_jac_g_body_struct = [[f'int {self.__counter} = 0;']]
    self.__eval_jac_g_body = [[f'int {self.__counter} = 0;']]
    self.__hess_non_zeros = S.Zero
    self.__eval_h_decl = set()
    self.__eval_h_body_struct = [[f'int {self.__counter} = 0;']]
    self.__eval_h_body = [[f'int {self.__counter} = 0;']]
    self.__finalize_solution_decl = set()
    self.__finalize_solution_body = [['std::FILE *fp;']]
    self.__user_functions = {}
    self.__user_data = {}
    self.__user_data_defs = []
    self.__user_data_decls = []

  @classmethod
  def __make_nlp_var(cls, name, indices, offset) :
    '''Возвращает объект-дескриптор для переменной.'''

    if indices :
      return cls.__BlockVar(name, indices, offset)
    else :
      return cls.__SingleVar(name, offset)

  def _check_bound(self, bound, indices) :
    '''Проверяем выражения верхней и нижней границы и начального значения.'''

    if not (set(get_outer_indices(bound)) <= set(indices)) :
      print('Bound indices are not in variable/constraint indices.')
      return False
    return (check_idx(bound)
            and check_symbols(bound, self.__user_data)
            and check_functions(bound, self.__user_functions)
            and check_for_exprs_with_int_limits(bound))

  def _check_expr(self, expr) :
    '''Проверка выражений ограничения и целевого функционала.'''

    return (check_idx(expr)
            and check_symbols(expr, self.__user_data, self.__variables)
            and check_functions(expr, self.__user_functions)
            and check_for_disjoint_block_vars(expr, self.__variables)
            and check_for_exprs_with_int_limits(expr))

  def _check_user_func_grad(self, grad, nargs) :
    '''Проверка градиента пользовательской функции.'''

    if len(grad) != nargs :
      print(f'Length of user function gradient should be equal to number of function arguments.')
      return False
    # Создадим набор аргументов для проверки элементов градиента
    args = tuple(RDummy() for _ in range(nargs))
    # Здесь будем считать эти аргументы известными
    user_data = {**self.__user_data, **{a : () for a in args}}
    for n in range(nargs) :
      partial = grad[n](*args)
      if not (check_idx(partial)
              and check_symbols(partial, user_data, self.__variables)
              and check_functions(partial, self.__user_functions)
              and check_for_disjoint_block_vars(partial, self.__variables)
              and check_for_exprs_with_int_limits(partial)
      ) :
        print(f'Bad expr for {n}-th element of gradient.')
        return False
      if not partial.free_symbols.issubset(args) :
        # В том числе запрещаем внешние индексы
        print(f'{n}-th element of gradient is incorrect: too many free symbols.')
        return False
    return True

  def add_user_data(self, name, indices = (), *, init = None, const = True) :
    '''Метод для добавления пользовательских данных --- переменных С++.

    Использование в выражениях задачи не конкретных значений, а имен C++ переменных
    может понадобиться, если:

    #. Планируется менять эти значения от запуска к запуску С++ программы. Например, начальные условия.
    #. Нужен массив значений, доступ к элементам которого осуществляется по индексам (аналогично переменным задачи).
       Например, в выражении верхней или нижней границы для переменной.

    Переменные, добавленные при помощи этого метода, будут объявлены в файле '<имя задачи>_user_decls.h'.

    Поддерживаются только вещественные пользовательские переменные-данные.
    
    :param str name: имя C++ переменной, нельзя использовать зарезервированные имена интерфейса IpOpt.
      Эти имена можно получить при помощи :meth:`reserved_names`.

    :param tuple indices: набор индексов для С++ массива или пустой набор для С++ переменной.
      Переданные здесь индексы определяют:

      #. Типы и порядок индексов массива (см. :class:`IdxType`).
      #. Размер массива.

      Требуется, чтобы каждый из индексов набора ``indices`` полностью проходил диапазон своего типа.
      Но, в отличие от переменной задачи, для массива не требуется уникальность типа каждого индекса в рамках набора.
      Преобразование многомерного массива в линейный ведется в стиле row-major (разворачивание по строкам),
      сдвиг нуля вычисляется автоматически, поэтому индексы могут пробегать любые диапазоны.
      Первый элемент всегда будет иметь нулевой индекс в массиве C++.
    :param init: список значений для массива или одно значение для переменной. Если None,
      то переменная будет только объявлена.

    :param bool const:

      #. если True, то переменная будет объявлена с модификатором const
      #. если False, то переменная будет объявлена без модификатора const

    :return: объект-дескриптор пользовательской переменной C++, который может использоваться в символьных выражениях SymPy.
      Для массива это производный класс `sympy.IndexedBase <https://docs.sympy.org/latest/modules/tensor/indexed.html#sympy.tensor.indexed.IndexedBase>`_,
      для переменной это `sympy.Symbol <https://docs.sympy.org/latest/modules/core.html?highlight=symbol#sympy.core.symbol.Symbol>`_.
      Этот объект создается с ``assumptions == {'real' : True}``.
      В выражениях задачи при пользовательском массиве наряду с индексами могут использоваться индексы со сдвигом.
      Для создания индекса со сдвигом используется класс :class:`ShiftedIdx`.

    >>> nlp = Nlp()
    >>> t1 = IdxType('t1', (1, 5))
    >>> t2 = IdxType('t2', (-8, -2))
    >>> i = t1('i')
    >>> j = t2('j')
    >>> f0 = nlp.add_user_data('f0', (i, j))
    >>> y0 = nlp.add_user_data('y0')
    >>> r0 = nlp.add_user_data('r0', (i, i), init = [0] * 25)
    >>> z0 = nlp.add_user_data('z0', init = 3, const = False)
    >>> i1 = t1('i1', (3, 5))
    >>> expr = z0 + f0[i1, j]

    .. seealso::
    
      :meth:`add_var`, :meth:`add_user_func`
    '''

    if self.__target_and_constrs_is_added :
      print('You should not add user data after IpOpt program has been generated!')
      raise RuntimeError
    if str(name) in self.reserved_names() :
      print(f'String representation of data "{name}" is reserved by IpOpt C++ interface. You shoud choose another name.')
      raise ValueError
    # Должна быть последовательность, а не просто итерируемый объект
    if indices :
      assert indices[0] == indices[0]
    indices = tuple(indices)
    if not check_idx(Tuple(*indices)) :
      raise ValueError
    if not is_full_range(indices) :
      print(f'Data "{name}" indices do not determine whole block with respect to index types.')
      raise ValueError
    size = block_size(indices)
    if indices :
      if init != None :
        if len(init) != size :
          print(f'Data "{name}" size is not equal to length of initial values')
          raise ValueError
      data = indexed_for_array_elem(name, indices, offset = S.Zero, real = True).base
    else :
      data = Symbol(name, real = True)
    if data in self.__user_data :
      print(f'User data {data} has already been added!')
      raise RuntimeError
    # Добавляем объявление переменной в файл '<имя задачи>_user_decls.h'
    self.__user_data_decls.append(
      'extern ' + ('const ' if const else '') + f'double {name}' + (f'[{size}]' if indices else '') + ';'
    ) 
    # Если переданы значения, добавляем определение переменной в файл '<имя задачи>.cpp'
    if init != None :
      self.__user_data_defs.append(
        ('const ' if const else '') + f'double {name}' + ('[] = {' + ', '.join(str(v) for v in init) + '}' if indices else f' = {init}') + ';'
      )
    # Сохраняем информацию о пользовательской переменной С++
    self.__user_data[data] = indices
    return data

  def add_user_func(self, name, nargs, grad = None) :
    '''Метод для добавления пользовательских функций (реализуемых на С++).

    Этот метод позволяет добавить функцию, которая должна использоваться в выражениях задачи,
    но ее реализация возможна только на С++.
    Например, если:
    
    #. Функция станет известной только после запуска программы на C++.
    #. Функция не может быть представлена в символьном виде.

    Объявление такой функции будет размещено в файле '<имя задачи>_user_decls.h',
    а ее реализация должна быть предоставлена пользователем, например в отдельном С++ модуле.

    Поддерживаются однозначные вещественные функции постоянного числа вещественных аргументов.

    :param str name: имя пользовательской функции, нельзя использовать зарезервированные имена интерфейса IpOpt.
      Эти имена можно получить при помощи :meth:`reserved_names`.

    :param int nargs: количество аргументов пользовательской функции (целое положительное число).

    :param grad: градиент пользовательской функции или None.
      Градиент должен быть представлен последовательностью, длина которой совпадает с числом аргументов функции (``nargs``).
      Элементами последовательности могут быть:

      #. Другая пользовательская функция.
      #. Вызываемый объект, который возвращает символьное выражение SymPy (
         `sympy.Lambda <https://docs.sympy.org/latest/modules/core.html?highlight=lambda#sympy.core.function.Lambda>`_,
         производные классы `sympy.Function <https://docs.sympy.org/latest/modules/core.html?highlight=function#sympy.core.function.Function>`_
         и т. п.).
         Выражение, полученное после вызова элемента градиента с некоторыми аргументами,
         должно удовлетворять требованиям к выражению ограничения задачи (подробности в :meth:`add_constr`) и дополнительно может
         содержать эти аргументы.

      Элемент градиента должен принимать столько же аргументов, сколько и сама функция: ``nargs``.
      Предполагается, что порядок аргументов у функции и ее градиента одинаков.

      Градиент можно не указывать (значение None).
      Если при аналитическом расчете градиента целевого функционала, якобиана или гессиана лагранжиана
      появится необходимость в дифференцировании функции, для которой градиент не указан, то будет выброшено исключение ``RuntimeError``.

    :return: объект-дескриптор пользовательской функции, который может использоваться в символьных выражениях SymPy.
      Это объект типа `sympy.UndefinedFunction <https://docs.sympy.org/latest/modules/core.html?highlight=function#sympy.core.function.Function>`_.

    >>> nlp = Nlp()
    >>> d2F = nlp.add_user_func('d2F', 1)
    >>> dF = nlp.add_user_func('dF', 1, (d2F,))
    >>> F = nlp.add_user_func('F', 1, (dF,))


    .. seealso::
    
      :meth:`add_user_data`
    '''

    if self.__target_and_constrs_is_added :
      print('You should not add user function after IpOpt program has been generated!')
      raise RuntimeError
    if nargs <= 0 :
      print(f'Number of user function "{name}" arguments should be positive.')
      raise ValueError
    if grad != None :
      if not self._check_user_func_grad(grad, nargs) :
        print(f'Bad gradient for function "{name}".')
        raise ValueError
    func = Function(name, nargs = nargs, real = True)
    if func in self.__user_functions :
      print(f'User function {func} has already been added!')
      raise RuntimeError
    # Добавляем объявление переменной в файл '<имя задачи>_user_decls.h'
    self.__user_data_decls.append(f'double {func}(' + ', '.join(['double'] * nargs) + ');')
    # Сохраняем информацию о пользовательской функции
    self.__user_functions[func] = grad
    return func

  def add_var(self, name, indices = (), *, starting_point, lower = None, upper = None, output_format = None) :
    '''Метод для добавления переменной в задачу.

    Поддерживаются только вещественные переменные.
    Они могут быть двух видов:

    #. Простые переменные. 
       Это обычные переменные, например :math:`y`.
    #. Блочные (индексированные) переменные.
       Это многомерный массив переменных задачи, с которыми в выражениях планируется работать
       посредством индексов, например :math:`f_{ijk}`.
       Нумерация переменных в блоке ведется в стиле row-major (разворачивание массива по строкам, как в С++), начиная с нуля,
       сдвиг вычисляется автоматически, поэтому индексы могут пробегать любые диапазоны.

    :param str name: имя переменной.

    :param tuple indices: набор индексов блочной переменой или пустой набор для простой переменной.
      Переданные здесь индексы определяют:

      #. Типы и порядок индексов блочной переменной.
      #. Размер блочной переменной (количество переменных в блоке).

      Тип каждого индекса должен быть уникальным в рамках набора.
      Это требование позволяет упростить процедуру аналитического вычисления якобиана и гессиана лагранжиана задачи.
      На данный момент не найдено примеров реальных задач,
      когда в выражениях задачи требуется отдельно использовать 'диагонали' блока переменных
      (например и :math:`f_{ij}`, и :math:`f_{ii}`).
      Для создания различных типов индексов используется метакласс :class:`IdxType`.
      Он создает тип с заданным именем, который является производным классом
      `sympy.Idx <https://docs.sympy.org/latest/modules/tensor/indexed.html#sympy.tensor.indexed.Idx>`_.
      Требуется, чтобы каждый из индексов набора ``indices`` полностью проходил диапазон своего типа.

    :param starting_point: символьное выражение SymPy, задающее начальное значение переменной.
      Используемые в нем индексы должны содержаться в ``indices``. Оно не должно зависеть от переменных задачи НЛП.
      Это выражение должно быть допустимым (подробности в :meth:`add_constr`).

    :param lower: символьное выражение SymPy, задающее нижнюю границу для переменной, или None.
      Используемые в нем индексы должны содержаться в ``indices``. Оно не должно зависеть от переменных задачи НЛП.
      Это выражение должно быть допустимым (подробности в :meth:`add_constr`).
      None означает отсутствие нижней границы.

    :param upper: символьное выражение SymPy, задающее верхнюю границу для переменной, или None.
      Используемые в нем индексы должны содержаться в ``indices``. Оно не должно зависеть от переменных задачи НЛП.
      Это выражение должно быть допустимым (подробности в :meth:`add_constr`).
      None означает отсутствие верхней границы.

    :param tuple(tuple, tuple) или None output_format: если не None, то после запуска IpOpt будет сгенерирован файл ``name``.out,
      содержащий финальное значение переменной. Это пара из наборов индексов,
      причем каждый из элементов этих наборов должен взаимно однозначно соответствовать по типу какому-либо индексу из ``indices``.
      Первый набор определяет строку выходного файла, второй набор --- столбец,
      на их пересечении будет финальное значение переменной с соответствующими индексами.
      Расчет строки и столбца аналогичен нумерации переменных в блоке: стиль row-major, сдвиг нуля вычисляется автоматически.

    :return: объект-дескриптор переменной, который может использоваться в символьных выражениях SymPy.
      Для простой переменной это производный класс `sympy.Symbol <https://docs.sympy.org/latest/modules/core.html?highlight=symbol#sympy.core.symbol.Symbol>`_,
      для блочной --- производный класс `sympy.IndexedBase <https://docs.sympy.org/latest/modules/tensor/indexed.html#sympy.tensor.indexed.IndexedBase>`_.
      Этот объект создается с ``assumptions == {'real' : True}``.
      При распечатывании этого объекта средствами SymPy, будет использоваться имя ``name``.
      В выражениях задачи при блочной переменной наряду с индексами могут использоваться индексы со сдвигом, например :math:`f_{i-1}`.
      Для создания индекса со сдвигом используется класс :class:`ShiftedIdx`.

    Пример использования:

    >>> nlp = Nlp()
    >>> t1 = IdxType('t1', (1, 5))
    >>> t2 = IdxType('t2', (-8, -2))
    >>> t3 = IdxType('t3', (-1, 4))
    >>> i = t1('i')
    >>> j = t2('j')
    >>> k = t3('k')
    >>> f0 = nlp.add_user_data('f0', (i,))
    >>> i1 = t1('i1', (3, 5))
    >>> f = nlp.add_var('f', (i, j, k),
                        starting_point = f0[i],
                        lower = S(5),
                        output_format = ((j, i1), (k)))
    >>> x = nlp.add_var('x',
                        starting_point = S.Zero,
                        upper = S.One,
                        output_format = ((), ()))
    >>> k1 = t3('k1', (0, 4))
    >>> sk1 = ShiftedIdx(k1, -1)
    >>> expr = x * (f[i1, j, k1] - f[i1, j, sk1])
    '''

    if self.__target_and_constrs_is_added :
      print('You should not add variable after IpOpt program has been generated!')
      raise RuntimeError
    if self.__vars_is_added :
      print('You should not add variable after constraint has been added!')
      raise RuntimeError
    # Должна быть последовательность, а не просто итерируемый объект
    if indices :
      assert indices[0] == indices[0]
    indices = tuple(indices)
    idx_types = get_types(indices)
    if len(set(idx_types)) != len(idx_types) :
      print(f'There are several indices of "{name}" with same type. Not supported.')
      raise ValueError
    if not check_idx(Tuple(*indices)) :
      print(f'Bad indices for variable "{name}".')
      raise ValueError
    if not is_full_range(indices) :
      print(f'Indices do not determine whole block with respect to index types for variable "{name}".')
      raise ValueError
    # None --- отсутствие ограничения. В этом случае используем константы из IpOpt
    if lower == None :
      lower = self.__nlp_lower_bound_inf
    if upper == None :
      upper = self.__nlp_upper_bound_inf
    if not (self._check_bound(lower, indices) and self._check_bound(upper, indices) and self._check_bound(starting_point, indices)) :
      print(f'Attempt to use bad expr for lower, upper or starting_point.')
      raise ValueError
    var = self.__make_nlp_var(name, indices, self.__var_offset)
    if var in self.__variables :
      print(f'Variable {var} has already been added!')
      raise RuntimeError
    if output_format != None :
      if len(output_format) != 2 :
        print(f'Bad output_format for variable "{var}".')
        raise ValueError
      format_indices = list(output_format[0] + output_format[1])
      if set(get_types(format_indices)) != set(idx_types) or len(format_indices) != len(indices) :
        print(f'output_format indices is not corresponds to variable "{var}" indices.')
        raise ValueError
      row_out, col_out = output_format
      # Расставляем индексы формата в порядке следовния при переменной
      if get_types(format_indices) != idx_types :
        order = {t : n for n, t in enumerate(idx_types)}
        for idx, t in zip(row_out, get_types(row_out)) :
          format_indices[order[t]] = idx
        for idx, t in zip(col_out, get_types(col_out)) :
          format_indices[order[t]] = idx
      # Добавляем код в тело функции finalize_solution() интерфейса IpOpt
      loop = [FPrint((var[format_indices] if format_indices else var,), '%f ', 'fp')]
      loop = wrap_in_loop(loop, col_out) + [FPrint((), '\\n', 'fp')]
      loop = wrap_in_loop(loop, row_out)
      body = [f'fp = std::fopen("{str(var)}.out", "w");'] + cxxcode(CodeBlock(*loop)).split('\n') + ['std::fclose(fp);']
      self.__finalize_solution_body.append(body)
      # Добавляем определения переменных C++, соответствующих индексам (для использования в циклах)
      self.__finalize_solution_decl.update(decl_for_indices(format_indices))
    # Храним зарегистрированные переменные в словаре OrderedDict:
    # ключ --- переменная
    # значение --- (позиция начала блока в одномерном массиве переменых задачи, tuple() или набор индексов блочной переменной)
    self.__variables[var] = (self.__var_offset, indices)
    # Добавляем код в тела функций get_bounds_info() и get_starting_point() интерфейса IpOpt
    self.__get_bounds_info_body.append(assign_by_indices((self.__x_l, self.__x_u), (lower, upper), indices, self.__var_offset))
    self.__get_starting_point_body.append(assign_by_indices((self.__x,), (starting_point,), indices, self.__var_offset))
    # Добавляем определения переменных C++, соответствующих индексам (для использования в циклах)
    self.__var_decl.update(decl_for_indices(indices))
    # Увеличиваем счетчик переменных на размер блока (1 для простой перменной)
    self.__var_offset += block_size(indices)
    return var

  def add_constr(self, constr, *, lower = None, upper = None) :
    '''Метод для добавления ограничения в задачу.

    Ограничение задачи НЛП должно иметь вид:

    .. math:: L \\le G \\le U,

    гдe
    :math:`G` --- это выражение ограничения,
    :math:`L` --- это выражение нижней границы,
    :math:`U` --- это выражение верхней границы.
    Все эти выражения должны быть скалярными.
    Нижняя и верхняя границы могут отсутствовать. 

    Если среди свободных символов выражения :math:`G` есть индексы, пробегающие больше одного значения,
    то :math:`G` задает целое семейство ограничений. Эти индексы будем называть внешними.
    Индексы суммирования будем называть внутренними.
    Выражение :math:`G` без внешних индексов будем считать семейством из одного выражения.

    :param constr: символьное выражение SymPy, задающее семейство ограничений.

    :param lower: символьное выражение SymPy, задающее нижнюю границу, или None.
      Используемые в нем индексы должны содержаться в множестве индексов семейства.
      Оно не должно зависеть от переменных задачи НЛП.
      None означает отсутствие нижней границы.

    :param upper: символьное выражение SymPy, задающее верхнюю границу, или None.
      Используемые в нем индексы должны содержаться в множестве индексов семейства.
      Оно не должно зависеть от переменных задачи НЛП.
      None означает отсутствие верхней границы.

    Все выражения задачи, в частности выражения ограничения и его нижней и верхней границ,
    должны удовлетворять следующим требованиям:

    #. Не должно быть различных индексов с одинаковым строковым представлением.
    #. Все индексы имеют длину __len__().
    #. Все типы индексов имеют атрибуты диапазона "start" и "end" и длину __len__().
    #. При блочных переменных и пользовательских массивах могут использоваться индексы и индексы со сдвигом (:class:`ShiftedIdx`).
       Другие выражения запрещены.
    #. Все свободные вхождения производных классов `sympy.Symbol <https://docs.sympy.org/latest/modules/core.html?highlight=symbol#sympy.core.symbol.Symbol>`_
       должны быть либо переменными задачи, либо пользовательскими данными.
    #. Все свободные вхождения производных классов `sympy.IndexedBase <https://docs.sympy.org/latest/modules/tensor/indexed.html#sympy.tensor.indexed.IndexedBase>`_
       должны быть либо переменными задачи, либо пользовательскими данными.
    #. Типы и количество индексов при переменных задачи должно соответствовать тому, что было указано при регистрации.
    #. Типы и количество индексов при пользовательских данных должно соответствовать тому, что было указано при регистрации.
    #. В выражении нет свободных символов кроме переменных задачи, пользовательских данных и индексов.
    #. Все вхождения `sympy.AppliedUndef <https://docs.sympy.org/latest/modules/core.html?highlight=function#sympy.core.function.Function>`_
       должны быть пользовательскими функциями от некоторых аргументов.
    #. В выражении перебор значений индексов
       может быть только в `sympy.Sum <https://docs.sympy.org/latest/modules/concrete.html?highlight=sum#sympy.concrete.summations.Sum>`_
       (в частности, запрещены произведения `sympy.Prod <https://docs.sympy.org/latest/modules/concrete.html?highlight=sum#sympy.concrete.products.Product>`_). 
    #. Требование постоянства структуры: индексы при различных вхождениях некоторой блочной переменной должны описывать непересекающиеся блоки.
       Два блока считаем непересекающимися, если:

       #. Они имеют одинаковые внешние индексы.
       #. Они не совпадают при подстановке вместо индексов любых значений из их диапазонов.
          Причем если в этих блоках есть один и тот же внешний индекс, то подстановка должна осуществляться одновременно.
          Например блоки, один из которых содержит внешний индекс :math:`i`, а другой --- :math:`i-1`, никогда не совпадут.


    >>> nlp = Nlp()
    >>> t1 = IdxType('t1', (1, 5))
    >>> t2 = IdxType('t2', (-8, -2))
    >>> i = t1('i')
    >>> j = t2('j')
    >>> g_L = nlp.add_user_data('g_L', (i,))
    >>> f = nlp.add_var('f', (i, j), starting_point = S.Zero)
    >>> x = nlp.add_var('x', starting_point = S.Zero)
    >>> nlp.add_constr(x * f[i, j],
                       lower = g_L[i],
                       upper = S.One)
    >>> nlp.add_constr(x**2, upper = S(5))

    .. seealso::
    
      :meth:`add_user_data`, :meth:`add_user_func`, :meth:`add_var`
    '''

    self.__vars_is_added = True
    if self.__target_and_constrs_is_added :
      print('You should not add constraint after IpOpt program has been generated!')
      raise RuntimeError
    if not self._check_expr(constr) :
      print('Attempt to add unallowable constraint.')
      raise ValueError
    # Делаем индексы суммирования уникальными
    constr = prepare_expr(constr)
    # Получаем индексы данного семейства ограничений
    indices = get_outer_indices(constr)
    # None --- отсутствие ограничения. В этом случае используем константы из IpOpt
    if lower == None :
      lower = self.__nlp_lower_bound_inf
    if upper == None :
      upper = self.__nlp_upper_bound_inf
    if not (self._check_bound(lower, indices) and self._check_bound(upper, indices)) :
      print(f'Attempt to use bad expr for lower or upper.')
      raise ValueError
    if constr in self.__constraints :
      print(f'Constraint {constr} has already been added!')
      raise RuntimeError
    # Храним ограничения в словаре OrderedDict:
    # ключ --- выражение ограничения
    # значение --- (позиция начала блока семейства ограничений, tuple() или набор индексов)
    self.__constraints[constr] = (self.__constr_offset, indices)
    # Добавляем код в тело функции get_bounds_info() интерфейса IpOpt
    self.__get_bounds_info_body.append(assign_by_indices((self.__g_l, self.__g_u), (lower, upper), indices, self.__constr_offset))
    # Добавляем определения переменных C++, соответствующих индексам (для использования в циклах)
    self.__constr_decl.update(decl_for_indices(indices))
    # Увеличиваем счетчик ограничений на размер семейства
    self.__constr_offset += block_size(indices)

  def set_obj(self, f) :
    '''Метод для добавления целевого функционала в задачу.

    :param f: символьное выражение SymPy, задающее целевой функционал.

    У целевого функционала не должно быть внешних индексов.
    Остальные требования такие же, как для выражений ограничений.

    .. seealso::
    
      :meth:`add_constr`
    '''

    self.__vars_is_added = True
    if self.__target_and_constrs_is_added :
      print('You should not set objective after IpOpt program has been generated!')
      raise RuntimeError
    if not self._check_expr(f) :
      print('Attempt to set unallowable objective.')
      raise ValueError
    # У целевого функционала не должно быть внешних индексов
    if get_outer_indices(f) :
      print('Objective should not have outer indices.')
      raise ValueError
    if self.__objective != None :
      print('Objective has already been set!')
      raise RuntimeError
    # Запоминаем целевой функционал
    self.__objective = prepare_expr(f)

  def __subs_user_func_derivative(self, expr) :
    '''Подставляем в выражжение частные производные пользовательских функций. '''

    derivatives = expr.atoms(Derivative)
    for d in derivatives :
      func, (arg, order), *other = d.args
      # Не должны появляться частные производные порядка больше 1
      assert not other and order == 1
      grad = self.__user_functions.get(func.func, None)
      if grad == None :
        print(f'Function "{func.func}" or its gradient is unknown!')
        raise RuntimeError
      partial = prepare_expr(grad[func.args.index(arg)](*func.args))
      expr = expr.subs(d, partial)
    return expr

  def __diff(self, expr, var, indices, occurrences) :
    '''Вычисляем производную выражения по переменной.'''

    # Учитываем дополнительные функции, заданные пользователем.
    # Для блочной переменной особая процедура.
    return self.__subs_user_func_derivative(diff_indexed(expr, var[indices], occurrences) if indices else diff(expr, var))

  def __fill_eval_f_body(self) :
    '''Генерируем код вычисления целевого функционала.'''

    self.__eval_f_body = [assign((self.__obj_value,), (self.__objective,))]

  def __fill_eval_g_body(self) :
    '''Генерируем код вычисления ограничений.'''
    for constr, (offset, indices) in self.__constraints.items() :
      self.__eval_g_body.append(assign_by_indices((self.__g,), (constr,), indices, offset))

  def __fill_eval_grad_f_body(self) :
    '''Генерируем код вычисления градиента целевого функционала.'''

    for var, (offset, indices) in self.__variables.items() :
      # Перебираем все вхождения данной переменной в целевом функционале для вычисления элементов градиента
      # Добавляем к ним полный блок переменной ``indices``, так как нулевые элементы градиента тоже нужны.
      for used_indices, occurrences in expr_var_indices(self.__objective, var, [block_copy(indices)]) :
        # Аналитическое дифференцирование
        partial = self.__diff(self.__objective, var, used_indices, occurrences)
        # Добавляем код в тела соответствующих функций интерфейса IpOpt
        self.__eval_grad_f_decl.update(decl_for_indices(used_indices))
        self.__eval_grad_f_body.append(assign_by_indices((self.__grad_f,), (partial,), used_indices, offset))

  def __fill_eval_jac_g_body(self) :
    '''Генерируем код вычисления якобиана ограничений.'''

    # Перебираем все ограничения --- проход по строкам
    for constr, (constr_offset, constr_indices) in self.__constraints.items() :
      # Перебираем все переменные --- проход по столбцам
      for var, (var_offset, var_indices) in self.__variables.items() :
        # Перебираем все вхождения данной переменной в этом ограничении для вычисления элементов градиента
        for used_indices, occurrences in expr_var_indices(constr, var) :
          # Аналитическое дифференцирование
          grad = self.__diff(constr, var, used_indices, occurrences)
          # Нам нужны только ненулевые элементы
          if grad != S.Zero :
            # used_indices и constr_indices, параметризуют положение ненулевого элемента в якобиане
            # Не используем множество, чтобы от запуска к запуску сохранялся порядок
            indices = constr_indices + tuple(idx for idx in get_master_idx(used_indices) if idx not in constr_indices)
            # Добавляем код в тела соответствующих функций интерфейса IpOpt
            self.__eval_jac_g_decl.update(decl_for_indices(indices))
            self.__eval_jac_g_body_struct.append(assign_by_counter((self.__iRow, self.__jCol),
                                                                          (constr_offset + offset_in_block(constr_indices),
                                                                           var_offset + offset_in_block(used_indices, part_of = var_indices)),
                                                                           indices, self.__counter))
            self.__eval_jac_g_body.append(assign_by_counter((self.__values,), (grad,), indices, self.__counter))
            # Увеличиваем счетчик ненулевых элементов якобиана на размер семейства
            self.__jac_non_zeros += block_size(indices) 

  def __fill_eval_h_body(self) :
    '''Генерируем код вычисления гессиана лагранжиана задачи.'''

    # Храним ненулевые элементы гессиана лагранжиана в словаре:
    # ключ --- смещение начала блока в гессиане: (смещение первой переменной дифференцирования, смещение второй переменной дифференцирования);
    # значение --- словарь:
    # ключ --- пара из индексов первой и второй переменной дифференцирования,
    # которая задает некоторый вложенный блок с ненулевыми элементами
    # значение --- аддитивная часть элементов этого вложенного блока (от целевого функционала или семейства ограничений)
    elems = defaultdict(lambda : defaultdict(lambda : S.Zero))
    # Вспомогательная функция, которая добавляет в словарь аддитивные части элементов, полученные при дифференцировании term.
    def process_term(term) :
      # Перебираем переменные для вычисления градиента term (проход по сторкам гессиана)
      for row_id, (row_offset, *_) in self.__variables.items() :
        # Проходим по всем вхождениям данной перменной для вычисления элементов градиента term
        for used_row_indices, row_occurrences in expr_var_indices(term, row_id) :
          # Аналитическое дифференцирование
          grad = self.__diff(term, row_id, used_row_indices, row_occurrences)
          # Нам нужны только ненулевые элементы
          if grad != S.Zero :
            # Перебираем переменные для второй операции дифференцирования (проход по столбцам гессиана)
            # Для IpOpt нужны блоки только из левой нижней части гессиана (гессиан будет симметричным в силу достаточной гладкости задачи)
            #for col_id, (col_offset, *_) in takewhile(lambda elem : elem[0] != row_id, self.__variables.items()) :
            for col_id, (col_offset, *_) in self.__variables.items() :
              hess_pos = (row_id, col_id)
              # Проходим по всем вхождениям данной перменной для вычисления элементов гессиана term
              for used_col_indices, col_occurrences in expr_var_indices(grad, col_id) :
                # Аналитическое дифференцирование
                hess = self.__diff(grad, col_id, used_col_indices, col_occurrences)
                # Запоминаем только ненулевые элементы
                if hess != S.Zero :
                  if col_id != row_id or sum(cmp_with_diag(used_row_indices, used_col_indices)[:2]) != 0 :
                    hess_block = (used_row_indices, used_col_indices)
                    elems[hess_pos][hess_block] += hess
              if col_id == row_id :
                break
    # Получаем вклад от целевого функционала
    process_term(self.__obj_factor * self.__objective)
    # Получаем вклад от ограничений
    for constr, (offset, indices) in self.__constraints.items() :
      term = indexed_for_array_elem(self.__lambda, indices, offset) * constr
      process_term(Sum(term, *indices) if indices else term)
    # Проходим по всем блокам гессиана
    for (row_id, col_id), hess_block in elems.items() :
      row_offset, row_indices = self.__variables[row_id]
      col_offset, col_indices = self.__variables[col_id]
      self.__eval_h_body_struct.append(cxxcode(Comment(f'{row_id}, {col_id}')).split('\n'))
      self.__eval_h_body.append(cxxcode(Comment(f'{row_id}, {col_id}')).split('\n'))
      # Для удобства обработки соединяем два набора индексов и получаем один длинный набор, описывающий некоторый многомерный параллелотоп
      # Если некоторые индексы из этого набора имеют одинаковые ведущие индексы, то от параллелотопа остается только "диагональ"
      # Так как у блочной переменной индексы различных типов,
      # то может быть не больше двух индексов с одинаковым ведущим индексом (по одному от каждого из исходных наборов)
      parts = [Part((row_id, col_id), indices[0] + indices[1], term) for indices, term in hess_block.items()]
      # Измельчаем части блока так, чтобы получить множество непересекающихся частей 
      parts = to_disjoint_parts(parts)
      for p in parts :
        continue_cond = False
        size = len(p)
        part_row_indices, part_col_indices = p.indices[:len(row_indices)], p.indices[len(row_indices):]
        row = row_offset + offset_in_block(part_row_indices, part_of = row_indices)
        col = col_offset + offset_in_block(part_col_indices, part_of = col_indices)
        if row_id == col_id :
          under, diag, over = cmp_with_diag(part_row_indices, part_col_indices)
          # Для диагональных блоков отбрасываем наддиагональные части, которые могли получиться при построении множества непересекающихся частей
          if under + diag == 0 :
            continue
          if over != 0 :
            continue_cond = col > row
            size -= over
        row_pre, row_body = assign_one(Element(self.__iRow, (self.__counter,)), row)
        assert not row_pre
        col_pre, col_body = assign_one(Element(self.__jCol, (self.__counter,)), col)
        assert not col_pre
        loop = p.generate_loop([*row_body, *col_body, PreIncrement(self.__counter)], continue_cond = continue_cond)
        self.__eval_h_body_struct.append(cxxcode(CodeBlock(*loop)).split('\n'))
        preambula, body = assign_one(Element(self.__values, (self.__counter,)), p.term, p.indices)
        body.append(PreIncrement(self.__counter))
        loop = p.generate_loop(body, continue_cond = continue_cond)
        self.__eval_h_body.append(cxxcode(CodeBlock(*preambula, *loop)).split('\n'))
        self.__eval_h_decl.update(decl_for_indices(p.indices))
        self.__hess_non_zeros += size

  def generate(self) :
    '''Запуск кодогенерации.

    После успешной кодогенерации в текущей директории будут получены файлы:

    #. '<имя задачи>.hpp',
    #. '<имя задачи>.cpp',
    #. '<имя задачи>_user_decls.h' (при необходимости).
    '''

    assert self.__vars_is_added
    self.__target_and_constrs_is_added = True

    common_names = {str(var) for var in self.__variables} & {str(data) for data in self.__user_data}
    if common_names :
      print(f'"{common_names}" are simultaneously used as variables and as user data')
      raise RuntimeError

    if self.__objective == None :
      raise RuntimeError('You should set objective function!')

    self.__fill_eval_f_body()
    self.__fill_eval_g_body()
    self.__fill_eval_grad_f_body()
    self.__fill_eval_jac_g_body()
    self.__fill_eval_h_body()

    self.__get_nlp_info_body = [assign((self.__n,), (S(self.__var_offset),)),
                                assign((self.__m,), (S(self.__constr_offset),)),
                                assign((self.__nnz_jac_g,), (S(self.__jac_non_zeros),)),
                                assign((self.__nnz_h_lag,), (S(self.__hess_non_zeros),))]

    generated_hpp = '''#ifndef __''' + self.__name.upper() + '''_HPP__
#define __''' + self.__name.upper() + '''_HPP__

#include "IpTNLP.hpp"

using namespace Ipopt;

class ''' + self.__name + ''': public TNLP
{
public:
   /** Default constructor */
   ''' + self.__name + '''();

   /** Default destructor */
   virtual ~''' + self.__name + '''();

   /**@name Overloaded from TNLP */
   //@{
   /** Method to return some info about the NLP */
   virtual bool get_nlp_info(
      Index&          n,
      Index&          m,
      Index&          nnz_jac_g,
      Index&          nnz_h_lag,
      IndexStyleEnum& index_style
   );

   /** Method to return the bounds for my problem */
   virtual bool get_bounds_info(
      Index   n,
      Number* x_l,
      Number* x_u,
      Index   m,
      Number* g_l,
      Number* g_u
   );

   /** Method to return the starting point for the algorithm */
   virtual bool get_starting_point(
      Index   n,
      bool    init_x,
      Number* x,
      bool    init_z,
      Number* z_L,
      Number* z_U,
      Index   m,
      bool    init_lambda,
      Number* lambda
   );

   /** Method to return the objective value */
   virtual bool eval_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number&       obj_value
   );

   /** Method to return the gradient of the objective */
   virtual bool eval_grad_f(
      Index         n,
      const Number* x,
      bool          new_x,
      Number*       grad_f
   );

   /** Method to return the constraint residuals */
   virtual bool eval_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Number*       g
   );

   /** Method to return:
    *   1) The structure of the jacobian (if "values" is NULL)
    *   2) The values of the jacobian (if "values" is not NULL)
    */
   virtual bool eval_jac_g(
      Index         n,
      const Number* x,
      bool          new_x,
      Index         m,
      Index         nnz_jac_g,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   /** Method to return:
    *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
    *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
    */
   virtual bool eval_h(
      Index         n,
      const Number* x,
      bool          new_x,
      Number        obj_factor,
      Index         m,
      const Number* lambda,
      bool          new_lambda,
      Index         nnz_h_lag,
      Index*        iRow,
      Index*        jCol,
      Number*       values
   );

   /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
   virtual void finalize_solution(
      SolverReturn               status,
      Index                      n,
      const Number*              x,
      const Number*              z_L,
      const Number*              z_U,
      Index                      m,
      const Number*              g,
      const Number*              lambda,
      Number                     obj_value,
      const IpoptData*           ip_data,
      IpoptCalculatedQuantities* ip_cq
   );
   //@}

};

#endif
'''

    user_declarations = '''#ifndef __''' + self.__name.upper() + '''_USER_DECLS_HPP__
#define __''' + self.__name.upper() + '''_USER_DECLS_HPP__
 
''' + '\n\n'.join(self.__user_data_decls) + '''

#endif
'''

    generated_cpp = '''#include "''' + self.__name + '''.hpp"

#include <cstdio>
#include <cmath>
''' + (f'\n#include "{self.__name}_user_decls.h"' if self.__user_data_decls else '') + '''

using namespace Ipopt;

// user data
''' + '\n'.join(self.__user_data_defs) + '''

// constructor
''' + self.__name + '''::''' + self.__name + '''()
{ }

// destructor
''' + self.__name + '''::~''' + self.__name + '''()
{ }

// [TNLP_get_nlp_info]
// returns the size of the problem
bool ''' + self.__name + '''::get_nlp_info(
   Index&          n,
   Index&          m,
   Index&          nnz_jac_g,
   Index&          nnz_h_lag,
   IndexStyleEnum& index_style
)
{

   ''' + '\n\n   '.join(['\n   '.join(block) for block in self.__get_nlp_info_body]) + '''

   // use the C style indexing (0-based)
   index_style = TNLP::C_STYLE;

   return true;
}

// [TNLP_get_bounds_info]
// returns the variable bounds
bool ''' + self.__name + '''::get_bounds_info(
   Index   n,
   Number* x_l,
   Number* x_u,
   Index   m,
   Number* g_l,
   Number* g_u
)
{
   ''' + '\n   '.join(sorted(self.__var_decl | self.__constr_decl)) + '''

   ''' + '\n\n   '.join(['\n   '.join(block) for block in self.__get_bounds_info_body]) + '''

   return true;
}

// [TNLP_get_starting_point]
// returns the initial point for the problem
bool ''' + self.__name + '''::get_starting_point(
   Index   n,
   bool    init_x,
   Number* x,
   bool    init_z,
   Number* z_L,
   Number* z_U,
   Index   m,
   bool    init_lambda,
   Number* lambda
)
{
   ''' + '\n   '.join(sorted(self.__var_decl)) + '''

   ''' + '\n\n   '.join(['\n   '.join(block) for block in self.__get_starting_point_body]) + '''

   return true;
}

// [TNLP_eval_f]
// returns the value of the objective function
bool ''' + self.__name + '''::eval_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number&       obj_value
)
{

   ''' + '\n\n   '.join(['\n   '.join(block) for block in self.__eval_f_body]) + '''

   return true;
}

// [TNLP_eval_g]
// return the value of the constraints: g(x)
bool ''' + self.__name + '''::eval_g(
   Index         n,
   const Number* x,
   bool          new_x,
   Index         m,
   Number*       g
)
{
   ''' + '\n   '.join(sorted(self.__constr_decl)) + '''

   ''' + '\n\n   '.join(['\n   '.join(block) for block in self.__eval_g_body]) + '''

   return true;
}

// [TNLP_eval_grad_f]
// return the gradient of the objective function grad_{x} f(x)
bool ''' + self.__name + '''::eval_grad_f(
   Index         n,
   const Number* x,
   bool          new_x,
   Number*       grad_f
)
{
   ''' + '\n   '.join(sorted(self.__eval_grad_f_decl)) + '''

   ''' + '\n\n   '.join(['\n   '.join(block) for block in self.__eval_grad_f_body]) + '''

   return true;
}

// [TNLP_eval_jac_g]
// return the structure or values of the Jacobian
bool ''' + self.__name + '''::eval_jac_g(
   Index         n,
   const Number* x,
   bool          new_x,
   Index         m,
   Index         nnz_jac_g,
   Index*        iRow,
   Index*        jCol,
   Number*       values
)
{
   ''' + '\n   '.join(sorted(self.__eval_jac_g_decl)) + '''

   if( values == NULL )
   {
      // return the structure of the Jacobian

      ''' + '\n\n      '.join(['\n      '.join(block) for block in self.__eval_jac_g_body_struct]) + '''

   }
   else
   {
      // return the values of the Jacobian of the constraints

      ''' + '\n\n      '.join(['\n      '.join(block) for block in self.__eval_jac_g_body]) + '''

   }

   return true;
}

// [TNLP_eval_h]
//return the structure or values of the Hessian
bool ''' + self.__name + '''::eval_h(
   Index         n,
   const Number* x,
   bool          new_x,
   Number        obj_factor,
   Index         m,
   const Number* lambda,
   bool          new_lambda,
   Index         nnz_h_lag,
   Index*        iRow,
   Index*        jCol,
   Number*       values
)
{
   ''' + '\n   '.join(sorted(self.__eval_h_decl)) + '''

   if( values == NULL )
   {
      // return the structure. This is a symmetric matrix, fill the lower left
      // triangle only.

      ''' + '\n\n      '.join(['\n      '.join(block) for block in self.__eval_h_body_struct]) + '''

   }
   else
   {
      // return the values. This is a symmetric matrix, fill the lower left
      // triangle only

      ''' + '\n\n      '.join(['\n      '.join(block) for block in self.__eval_h_body]) + '''

   }

   return true;
}

// [TNLP_finalize_solution]
void ''' + self.__name + '''::finalize_solution(
   SolverReturn               status,
   Index                      n,
   const Number*              x,
   const Number*              z_L,
   const Number*              z_U,
   Index                      m,
   const Number*              g,
   const Number*              lambda,
   Number                     obj_value,
   const IpoptData*           ip_data,
   IpoptCalculatedQuantities* ip_cq
)
{
   ''' + '\n   '.join(sorted(self.__finalize_solution_decl)) + '''

   ''' + '\n\n   '.join(['\n   '.join(block) for block in self.__finalize_solution_body]) + '''
}
'''

    with open(self.__name + '.hpp', 'w') as f:
      f.write(generated_hpp)

    if self.__user_data_decls :
      with open(self.__name + '_user_decls.h', 'w') as f:
        f.write(user_declarations)

    with open(self.__name + '.cpp', 'w') as f:
      f.write(generated_cpp)

  @classmethod
  def _self_test(cls) :
    from sympy import Indexed, Sum, Product, Lambda, sin
    from sympy2ipopt.utils.code_utils import cxxcode
    from sympy2ipopt.utils.expr_utils import to_expr
    from sympy2ipopt.utils.test_utils import check_value_error, check_runtime_error, check_type_error, renum_dummy

    nlp = cls()

    y = Symbol('y')

    r = IndexedBase('r')
    f = IndexedBase('f')
    p = IndexedBase('p')

    t1 = IdxType('t1', (1, 3))
    t2 = IdxType('t2', (2,5))

    i1 = t1('i1')
    i2 = t2('i2')

    j1 = t1('j1', (2, 3))
    j2 = t2('j2', (2, 4))

    sj1 = ShiftedIdx(j1, -1)
    sj2 = ShiftedIdx(j2, 1)


    #assert
    r0 = nlp.add_user_data('r0', (i1, i2))
    assert r0 == IndexedBaseWithOffset('r0', shape = (3, 4), offset = -6, real = True)
    p0 = nlp.add_user_data('p0', (i2,), init = [4, 3, 2, 1])
    assert p0 == IndexedBaseWithOffset('p0', shape = (4,), offset = -2, real = True)
    fU = nlp.add_user_data('fU', (), init = S.One)
    assert fU == Symbol('fU', real = True)
    fS = nlp.add_user_data('fS', ())
    t3 = IdxType('t3', (0, 8))
    p1 = nlp.add_user_data('p1', (t3('i3'),))
    c0 = nlp.add_user_data('c0', (i1, i2)) 

    i3 = IdxType('t3', (-5,3))('i3')
    i4 = IdxType('t4', (-9,-4))('i4')
    g0 = IndexedBase('g0')
    assert nlp._check_bound(r0[i1, i2]*g0[i3], (i1, i2, i3, i4)) == False
    g0 = nlp.add_user_data('g0', (i2,))
    assert nlp._check_bound(r0[i1, i2]*g0[i3], (i1, i2, i3, i4)) == False
    g0 = nlp.add_user_data('g0', (i3,))
    assert nlp._check_bound(r0[i1, i2]*g0[i3], (i1, i2, i3, i4)) == True
    assert nlp._check_bound(r0[i1, i2]*g0[i3], (i1, i2, i4)) == False


    r = nlp.add_var('r', (i1, i2), starting_point = S(10), lower = S.Zero, upper = r0[i1, i2])
    y = nlp.add_var('y', (), starting_point = S(10), lower = S(5), upper = None)
    f = nlp.add_var('f', (i1,), starting_point = fS, lower = None, upper = fU)
    p = nlp.add_var('p', (i1, i2), starting_point = S(8), lower = p0[i2], upper = S(10))
    check_value_error(p.func, 'p1', shape = (3, 1))
    assert p.func('p2') == nlp.__BlockVar('p2', (i1, i2), 16)
    assert p.func('p2', (3, 4)) == nlp.__BlockVar('p2', (i1, i2), 16)
    assert p.func('p2', (3, 4)) == nlp.__BlockVar('p2', (i1, i2), 1)
    assert cxxcode(p.func('p2')[i1, i2]) == 'x[10 + 4*i1 + i2]'
    assert y.func('y1') == nlp.__SingleVar('y1', 12)
    assert y.func('y1') == nlp.__SingleVar('y1', 1)
    assert cxxcode(y.func('y2')) == 'x[12]'
    check_runtime_error(nlp.add_var, 'r', (i1, i2), starting_point = S.Zero, lower = S.Zero, upper = S.One)
    check_runtime_error(nlp.add_var, 'y', (), starting_point = S.Zero, lower = S.Zero, upper = S.One)
    check_value_error(nlp.add_var, 'r', (i1, j2), starting_point = S.Zero, lower = S.Zero, upper = S.One)
    check_value_error(nlp.add_var, 'r', (i1, i1), starting_point = S.Zero, lower = S.Zero, upper = S.One)
    check_value_error(nlp.add_var, 'r', (i1, i2), starting_point = r0[j1, j2], lower = S.Zero, upper = S.One)
    check_value_error(nlp.add_var, 'r', (i1, i2), starting_point = S.Zero, lower = r0[j1, j2], upper = S.One)
    check_value_error(nlp.add_var, 'r', (i1, i2), starting_point = S.Zero, lower = S.Zero, upper = r0[j1, j2])
    check_value_error(nlp.add_var, 'r', (i1, sj1), starting_point = S.Zero, lower = S.Zero, upper = S.One)

    assert nlp.__variables == OrderedDict([(r, (0, (i1, i2))), (y, (12, ())), (f, (13, (i1,))), (p, (16, (i1, i2)))])
    assert nlp.__var_offset == 28
    assert nlp.__var_decl == {'int i1;', 'int i2;'}
    assert [to_expr(var) for var in nlp.__variables] == [IndexedBaseWithOffset(nlp.__x, (3, 4), -6),
                                   Indexed(IndexedBaseWithOffset(nlp.__x, (1,), 12), 0),
                                   IndexedBaseWithOffset(nlp.__x, (3,), 12),
                                   IndexedBaseWithOffset(nlp.__x, (3, 4), 10)]
    assert nlp.__get_bounds_info_body == [[
                                           'for (i1 = 1; i1 < 4; i1 += 1) {',
                                           '   for (i2 = 2; i2 < 6; i2 += 1) {',
                                           '      x_l[-6 + 4*i1 + i2] = 0;',
                                           '      x_u[-6 + 4*i1 + i2] = r0[-6 + 4*i1 + i2];',
                                           '   };',
                                           '};'
                                          ], [
                                           'x_l[12] = 5;',
                                           'x_u[12] = 1.0e+19;'
                                          ], [
                                           'for (i1 = 1; i1 < 4; i1 += 1) {',
                                           '   x_l[12 + i1] = -1.0e+19;',
                                           '   x_u[12 + i1] = fU;',
                                           '};'
                                          ], [
                                           'for (i1 = 1; i1 < 4; i1 += 1) {',
                                           '   for (i2 = 2; i2 < 6; i2 += 1) {',
                                           '      x_l[10 + 4*i1 + i2] = p0[-2 + i2];',
                                           '      x_u[10 + 4*i1 + i2] = 10;',
                                           '   };',
                                           '};'
                                          ]]
    get_bounds_info_body_len = len(nlp.__get_bounds_info_body)


    assert nlp.__get_starting_point_body == [[
                                              'for (i1 = 1; i1 < 4; i1 += 1) {',
                                              '   for (i2 = 2; i2 < 6; i2 += 1) {',
                                              '      x[-6 + 4*i1 + i2] = 10;',
                                              '   };',
                                              '};'
                                             ], [
                                              'x[12] = 10;'
                                             ], [
                                              'for (i1 = 1; i1 < 4; i1 += 1) {',
                                              '   x[12 + i1] = fS;',
                                              '};'
                                             ], [
                                              'for (i1 = 1; i1 < 4; i1 += 1) {',
                                              '   for (i2 = 2; i2 < 6; i2 += 1) {',
                                              '      x[10 + 4*i1 + i2] = 8;',
                                              '   };',
                                              '};'
                                             ]]

    assert nlp._check_expr(y**3)
    check_runtime_error(nlp.add_var, 'y', (), starting_point = S.Zero, lower = S.Zero, upper = S.Zero)
    def test_order() :
      nlp = cls()
      nlp.add_constr(S.One, lower = S.One, upper = S.One)
      check_runtime_error(nlp.add_var, 'y', (), starting_point = S.Zero, lower = S.Zero, upper = S.Zero)
    test_order()
    def test_order() :
      nlp = cls()
      nlp.set_obj(S.One)
      check_runtime_error(nlp.add_var, 'y', (), starting_point = S.Zero, lower = S.Zero, upper = S.Zero)
    test_order()
    def test_order() :
      nlp = cls()
      try :
        nlp.set_obj(S.One)
        nlp.add_constr(S.One, lower = S.One, upper = S.One)
      except :
        assert False
    test_order()

    nlp.add_constr(y**3, lower = S(30), upper = None)
    nlp.add_constr(r[j1, j2]**f[sj1], lower = None, upper = S(100))
    nlp.add_constr((r[sj1, i2] + p[sj1, i2])**2, lower = S.Zero, upper = c0[sj1, i2])
    nlp.add_constr(Sum(r[j1, i2]**2, i2) + p[j1, j2], lower = S(10), upper = S(100))

    check_value_error(nlp.add_constr, Product(p[i1, i2], i1), lower = S.Zero, upper = S.One)
    check_value_error(nlp.add_constr, p[i1, i2], lower = c0[j1, j2], upper = S.One)
    check_value_error(nlp.add_constr, p[i1, i2], lower = S.One, upper = c0[j1, j2])
    check_value_error(nlp.add_constr, p[i1, i2], lower = S.One, upper = IndexedBase('a0')[j1, j2])

    assert nlp.__constraints == OrderedDict([(y**3, (0, ())), (r[j1, j2]**f[sj1], (1, (j1, j2))), ((p[sj1, i2] + r[sj1, i2])**2, (7, (i2, j1))), (p[j1, j2] + Sum(r[j1, i2]**2, (i2, 2, 5)), (15, (j1, j2)))])
    assert nlp.__constr_offset == 21
    assert nlp.__constr_decl == {'int j1;', 'int j2;', 'int i2;'}

    assert nlp.__get_bounds_info_body[get_bounds_info_body_len:] == [[
                                                                      'g_l[0] = 30;',
                                                                      'g_u[0] = 1.0e+19;'
                                                                     ], [
                                                                      'for (j1 = 2; j1 < 4; j1 += 1) {',
                                                                      '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                                                      '      g_l[-7 + 3*j1 + j2] = -1.0e+19;',
                                                                      '      g_u[-7 + 3*j1 + j2] = 100;',
                                                                      '   };',
                                                                      '};'
                                                                     ], [
                                                                      'for (i2 = 2; i2 < 6; i2 += 1) {',
                                                                      '   for (j1 = 2; j1 < 4; j1 += 1) {',
                                                                      '      g_l[1 + 2*i2 + j1] = 0;',
                                                                      '      g_u[1 + 2*i2 + j1] = c0[-10 + 4*j1 + i2];',
                                                                      '   };',
                                                                      '};'
                                                                      ], [
                                                                       'for (j1 = 2; j1 < 4; j1 += 1) {',
                                                                       '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                                                       '      g_l[7 + 3*j1 + j2] = 10;',
                                                                       '      g_u[7 + 3*j1 + j2] = 100;',
                                                                       '   };',
                                                                       '};'
                                                                     ]]

    l1 = t1('l1')
    l2 = t2('l2')
    check_value_error(nlp.set_obj, Sum(r[i1, i2], i2))
    nlp.set_obj(Sum(r[i1, i2]**2 + f[i1]**2, i1, i2) + Sum(p[l1, l2]**2, l1, l2))

    F = nlp.add_user_func('F', 3)
    assert F in nlp.__user_functions
    assert nlp.__user_functions[F] == None
    class MyFunc(Function) :
      nargs = 2
      @classmethod
      def eval(cls, arg1, arg2) :
        return arg1*arg2 + 5
    y1 = Symbol('y1')
    y2 = Symbol('y2')
    y3 = Symbol('y3')
    check_type_error(nlp.add_user_func, 'G', 3, (Lambda(y1, y1**2), F, F))
    check_type_error(nlp.add_user_func, 'G', 3, (Lambda((y1, y2, y3), y1**2), MyFunc, F))
    class MyFunc(Function) :
      nargs = 3
      @classmethod
      def eval(cls, arg1, arg2, arg3) :
        return arg1 * arg2 + 5
    P = Function('P', nargs = 2)
    check_type_error(nlp.add_user_func, 'G', 3, (Lambda((y1, y2, y3), y1**2), MyFunc, P))
    P1 = Function('P1', nargs = 3)
    check_value_error(nlp.add_user_func, 'G', 3, (Lambda((y1, y2, y3), y1**2), MyFunc, P1))
    check_value_error(nlp.add_user_func, 'G', 3, (Lambda((y1, y2, y3), y1**2 + y), MyFunc, F))
    check_value_error(nlp.add_user_func, 'G', 3, (Lambda((y1, y2, y3), y1**2), MyFunc))
    check_value_error(nlp.add_user_func, 'G', 3, (Lambda((y1, y2, y3), y1**2 * i1), MyFunc, F))
    G = nlp.add_user_func('G', 3, (Lambda((y1, y2, y3), y1**2), MyFunc, F))
    assert G in  nlp.__user_functions
    assert nlp.__user_functions[G] == (Lambda((y1, y2, y3), y1**2), MyFunc, F)

    assert nlp.__subs_user_func_derivative(Derivative(G(y1, y2, y3), y2)) == y1 * y2 + 5
    assert nlp.__subs_user_func_derivative(Derivative(G(y1, y2, y3), y1)) == y1**2
    assert nlp.__subs_user_func_derivative(Derivative(G(y1, y2, y3), y3)) == F(y1, y2, y3)
    check_runtime_error(nlp.__subs_user_func_derivative, Derivative(F(y1, y2, y3), y3))
    assert nlp.__subs_user_func_derivative(8 + y**Derivative(G(y1, y2, y3), y3) * Derivative(G(y1, y2, y3), y1)
                                           + sin(Derivative(G(y1, y2, y3), y2))) == S(8) + y**F(y1, y2, y3) * y1**2 + sin(y1 * y2 + 5)

    assert nlp.__diff(G(y1, y2, y3) + y2**2, y3, (), ()) == F(y1, y2, y3)
    assert nlp.__diff(G(y1, r[i1, i2], y3)**2 + y2**2, r, (i1, i2), (r[i1, i2],)) == 2 * ((y1 * r[i1, i2] + 5) * G(y1, r[i1, i2], y3))

    assert cxxcode(G(y1, r[i1, i2], y3)**2 + y2**2) == 'std::pow(y2, 2) + std::pow(G(y1, x[-6 + 4*i1 + i2], y3), 2)'

    assert nlp.__vars_is_added
    assert not nlp.__target_and_constrs_is_added
    nlp.__target_and_constrs_is_added = True
    check_runtime_error(nlp.add_constr, S.Zero, lower = S.Zero, upper = S.Zero)
    check_runtime_error(nlp.set_obj, S.Zero)
    check_runtime_error(nlp.add_user_func, 'F', 1)
    check_runtime_error(nlp.add_user_data, 'F', 1)

    assert cxxcode(y + r[i1, i2] + f[i1] + p[j1, sj2]) == 'x[12] + x[12 + i1] + x[-6 + 4*i1 + i2] + x[11 + 4*j1 + j2]'

    nlp.__fill_eval_f_body()
    assert renum_dummy(nlp.__eval_f_body) == [[
                                  'int _Dummy_3;',
                                  'int _Dummy_4;',
                                  'int _Dummy_5;',
                                  'int _Dummy_6;',
                                  'double _Dummy_1 = 0.0;',
                                  'for (_Dummy_3 = 1; _Dummy_3 < 4; _Dummy_3 += 1) {',
                                  '   for (_Dummy_4 = 2; _Dummy_4 < 6; _Dummy_4 += 1) {',
                                  '      _Dummy_1 += std::pow(x[10 + 4*_Dummy_3 + _Dummy_4], 2);',
                                  '   };',
                                  '};',
                                  'double _Dummy_2 = 0.0;',
                                  'for (_Dummy_5 = 1; _Dummy_5 < 4; _Dummy_5 += 1) {',
                                  '   for (_Dummy_6 = 2; _Dummy_6 < 6; _Dummy_6 += 1) {',
                                  '      _Dummy_2 += std::pow(x[12 + _Dummy_5], 2) + std::pow(x[-6 + 4*_Dummy_5 + _Dummy_6], 2);',
                                  '   };',
                                  '};',
                                  'obj_value = _Dummy_1 + _Dummy_2;'
                                 ]] 
    nlp.__fill_eval_g_body()
    assert renum_dummy(nlp.__eval_g_body) == [[
                                  'g[0] = std::pow(x[12], 3);'
                                 ], [
                                  'for (j1 = 2; j1 < 4; j1 += 1) {',
                                  '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                  '      g[-7 + 3*j1 + j2] = std::pow(x[-6 + 4*j1 + j2], x[11 + j1]);',
                                  '   };',
                                  '};'
                                 ], [
                                  'for (i2 = 2; i2 < 6; i2 += 1) {',
                                  '   for (j1 = 2; j1 < 4; j1 += 1) {',
                                  '      g[1 + 2*i2 + j1] = std::pow(x[-10 + 4*j1 + i2] + x[6 + 4*j1 + i2], 2);',
                                  '   };',
                                  '};'
                                 ], [
                                  'int _Dummy_2;',
                                  'double _Dummy_1;',
                                  'for (j1 = 2; j1 < 4; j1 += 1) {',
                                  '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                  '      _Dummy_1 = 0.0;',
                                  '      for (_Dummy_2 = 2; _Dummy_2 < 6; _Dummy_2 += 1) {',
                                  '         _Dummy_1 += std::pow(x[-6 + 4*j1 + _Dummy_2], 2);',
                                  '      };',
                                  '      g[7 + 3*j1 + j2] = _Dummy_1 + x[10 + 4*j1 + j2];',
                                  '   };',
                                  '};'
                                 ]]

    nlp.__fill_eval_grad_f_body()
    assert renum_dummy(nlp.__eval_grad_f_body) == [[
                                       'for (_Dummy_1 = 1; _Dummy_1 < 4; _Dummy_1 += 1) {',
                                       '   for (_Dummy_2 = 2; _Dummy_2 < 6; _Dummy_2 += 1) {',
                                       '      grad_f[-6 + 4*_Dummy_1 + _Dummy_2] = 2*x[-6 + 4*_Dummy_1 + _Dummy_2];',
                                       '   };',
                                       '};'
                                      ], [
                                       'grad_f[12] = 0;'
                                      ], [
                                       'for (_Dummy_8 = 1; _Dummy_8 < 4; _Dummy_8 += 1) {',
                                       '   grad_f[12 + _Dummy_8] = 8*x[12 + _Dummy_8];',
                                       '};'
                                      ], [
                                       'for (_Dummy_15 = 1; _Dummy_15 < 4; _Dummy_15 += 1) {',
                                       '   for (_Dummy_16 = 2; _Dummy_16 < 6; _Dummy_16 += 1) {',
                                       '      grad_f[10 + 4*_Dummy_15 + _Dummy_16] = 2*x[10 + 4*_Dummy_15 + _Dummy_16];',
                                       '   };',
                                       '};'
                                      ]]
    assert renum_dummy(nlp.__eval_grad_f_decl) == {'int _Dummy_1;', 'int _Dummy_15;', 'int _Dummy_2;', 'int _Dummy_8;', 'int _Dummy_16;'}

    nlp.__fill_eval_jac_g_body()
    assert nlp.__eval_jac_g_body == [[
                                      'int counter = 0;'
                                     ], [
                                      'values[counter] = 3*std::pow(x[12], 2);',
                                      '++(counter);'
                                     ], [
                                      'for (j1 = 2; j1 < 4; j1 += 1) {',
                                      '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                      '      values[counter] = std::pow(x[-6 + 4*j1 + j2], x[11 + j1])*x[11 + j1]/x[-6 + 4*j1 + j2];',
                                      '      ++(counter);',
                                      '   };',
                                      '};'
                                     ], [
                                      'for (j1 = 2; j1 < 4; j1 += 1) {',
                                      '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                      '      values[counter] = std::pow(x[-6 + 4*j1 + j2], x[11 + j1])*std::log(x[-6 + 4*j1 + j2]);',
                                      '      ++(counter);',
                                      '   };',
                                      '};'
                                     ], [
                                      'for (i2 = 2; i2 < 6; i2 += 1) {',
                                      '   for (j1 = 2; j1 < 4; j1 += 1) {',
                                      '      values[counter] = 2*x[-10 + 4*j1 + i2] + 2*x[6 + 4*j1 + i2];',
                                      '      ++(counter);',
                                      '   };',
                                      '};'
                                     ], [
                                      'for (i2 = 2; i2 < 6; i2 += 1) {',
                                      '   for (j1 = 2; j1 < 4; j1 += 1) {',
                                      '      values[counter] = 2*x[-10 + 4*j1 + i2] + 2*x[6 + 4*j1 + i2];',
                                      '      ++(counter);',
                                      '   };',
                                      '};'
                                     ], [
                                      'for (j1 = 2; j1 < 4; j1 += 1) {',
                                      '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                      '      for (i2 = 2; i2 < 6; i2 += 1) {',
                                      '         values[counter] = 2*x[-6 + 4*j1 + i2];',
                                      '         ++(counter);',
                                      '      };',
                                      '   };',
                                      '};'
                                     ], [
                                      'for (j1 = 2; j1 < 4; j1 += 1) {',
                                      '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                      '      values[counter] = 1;',
                                      '      ++(counter);',
                                      '   };',
                                      '};'
                                     ]]

    assert nlp.__eval_jac_g_body_struct == [[
                                             'int counter = 0;'
                                            ], [
                                             'iRow[counter] = 0;',
                                             'jCol[counter] = 12;',
                                             '++(counter);'
                                            ], [
                                             'for (j1 = 2; j1 < 4; j1 += 1) {',
                                             '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                             '      iRow[counter] = -7 + 3*j1 + j2;',
                                             '      jCol[counter] = -6 + 4*j1 + j2;',
                                             '      ++(counter);',
                                             '   };',
                                             '};'
                                            ], [
                                             'for (j1 = 2; j1 < 4; j1 += 1) {',
                                             '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                             '      iRow[counter] = -7 + 3*j1 + j2;',
                                             '      jCol[counter] = 11 + j1;',
                                             '      ++(counter);',
                                             '   };',
                                             '};'
                                            ], [
                                             'for (i2 = 2; i2 < 6; i2 += 1) {',
                                             '   for (j1 = 2; j1 < 4; j1 += 1) {',
                                             '      iRow[counter] = 1 + 2*i2 + j1;',
                                             '      jCol[counter] = -10 + 4*j1 + i2;',
                                             '      ++(counter);',
                                             '   };',
                                             '};'
                                            ], [
                                             'for (i2 = 2; i2 < 6; i2 += 1) {',
                                             '   for (j1 = 2; j1 < 4; j1 += 1) {',
                                             '      iRow[counter] = 1 + 2*i2 + j1;',
                                             '      jCol[counter] = 6 + 4*j1 + i2;',
                                             '      ++(counter);',
                                             '   };',
                                             '};'
                                            ], [
                                             'for (j1 = 2; j1 < 4; j1 += 1) {',
                                             '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                             '      for (i2 = 2; i2 < 6; i2 += 1) {',
                                             '         iRow[counter] = 7 + 3*j1 + j2;',
                                             '         jCol[counter] = -6 + 4*j1 + i2;',
                                             '         ++(counter);',
                                             '      };',
                                             '   };',
                                             '};'
                                            ], [
                                             'for (j1 = 2; j1 < 4; j1 += 1) {',
                                             '   for (j2 = 2; j2 < 5; j2 += 1) {',
                                             '      iRow[counter] = 7 + 3*j1 + j2;',
                                             '      jCol[counter] = 10 + 4*j1 + j2;',
                                             '      ++(counter);',
                                             '   };',
                                             '};'
                                            ]]

    assert nlp.__eval_jac_g_decl == {'int i2;', 'int j2;', 'int j1;'}

    assert nlp.__jac_non_zeros == 59

    #nlp1 = cls()
    #k1 = t1('k1')
    #k2 = t2('k2')
    #a = IndexedBase('a')
    #b = IndexedBase('b')
    #nlp1.add_var(a[i1, i2], starting_point = S.Zero, lower = None, upper = S.One)
    #nlp1.add_var(b[i1, i2], starting_point = S.Zero, lower = None, upper = S.One)
    #elems = {(a, b) : {((j1, i2), (sj1, k2)) : a[j1, i2] * b[sj1, k2], ((i1, j2), (k1, sj2)) : 2 * a[i1, j2] * b[k1, sj2], ((j1, j2), (sj1, sj2)) : 3 * a[j1, j2] * b[sj1, sj2], ((j1, j2), (j1, j2)) : 4 * a[j1, j2] * b[j1, j2], ((i1, j2), (i1, j2)) : 5 * a[i1, j2] * b[i1, j2]}}
    #elems = {(a, b) : {((), (j1, i2, sj1, k2)) : a[j1, i2] * b[sj1, k2], ((), (i1, j2, k1, sj2)) : 2 * a[i1, j2] * b[k1, sj2], ((), (j1, j2, sj1, sj2)) : 3 * a[j1, j2] * b[sj1, sj2], ((), (j1, j2, j1, j2)) : 4 * a[j1, j2] * b[j1, j2], ((), (i1, j2, i1, j2)) : 5 * a[i1, j2] * b[i1, j2]}}
    #hess_elems = nlp1.__prepare_hess_elems(elems)
    #pprint(hess_elems)
    #nlp1.__process_loops(hess_elems)
    #print('\n\n      '.join(['\n      '.join(block) for block in nlp1.__eval_h_body_struct]))
    #print('\n\n      '.join(['\n      '.join(block) for block in nlp1.__eval_h_body]))

    #nlp.__fill_eval_h_body()

    #nlp.generate()

if __name__ == "__main__" :
  Nlp._self_test()

  print('ALL TESTS HAVE BEEN PASSED!!!')
