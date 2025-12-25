#!/bin/python3

from sympy import Function, Pow, S, Sum, Range, cse
from sympy.core.sympify import _sympify
from sympy.codegen.ast import Element, For, AddAugmentedAssignment, Assignment, CodeBlock, Declaration, Variable, Token, Print
from sympy.codegen.cnodes import PreIncrement
from functools import reduce
from sympy.printing.cxx import CXX11CodePrinter

from sympy2ipopt.idx_type import IdxType
from sympy2ipopt.indexed_base_with_offset import IndexedBaseWithOffset
from sympy2ipopt.utils.idx_utils import IDummy, get_master_idx, get_shifts, block_shape, block_size
from sympy2ipopt.utils.expr_utils import to_expr, get_sums, RDummy, get_outer_indices

class EmptyOperator(Token) :
  ''' Пустой оператор языка С: ';'. '''

empty_operator_ = EmptyOperator()

class If(Token) :
  __slots__ = ('condition', 'body', 'else_branch')
  _fields = __slots__
  defaults = {'else_branch' : []}
  _construct_condition = staticmethod(_sympify)
  @classmethod
  def _construct_body(cls, itr):
    if isinstance(itr, CodeBlock):
      return itr
    else:
      return CodeBlock(*itr)
  @classmethod
  def _construct_else_branch(cls, itr):
    if isinstance(itr, CodeBlock):
      return itr
    else:
      return CodeBlock(*itr)

class FPrint(Print) :
  defaults = {'print_args' : (),'format_string': '', 'file': 'stdout'}

class _SpecificCodePrinter(CXX11CodePrinter) :
  ''' CXX11CodePrinter с переопределенными _print_* для Pow, EmptyOperator, If, Stream и FPrint. '''
  def _print_Pow(self, expr) :
    try :
      power = int(expr.exp)
    except TypeError :
      pass
    else :
      u_power = abs(power)
      if power == expr.exp and u_power > 1 and u_power < 20:
        ret = self._print(expr.base)
        if not (expr.base.is_Atom or isinstance(expr.base, Function)) :
          ret = '(' + ret + ')'
        ret = '(' + '*'.join([ret]*u_power) + ')'
        if power < 0 :
          ret = f'(1/{ret})'
        return ret
    return super()._print_Pow(expr)
  def _print_EmptyOperator(self, expr) :
    return ';'
  def _print_If(self, expr) :
    if expr.condition == True :
      if expr.body.args :
        return self._print(expr.body)
      else :
        return self._print(empty_operator_)
    elif expr.condition == False :
      if expr.else_branch.args :
        return self._print(expr.else_branch)
      else :
        return self._print(empty_operator_)
    else :
      condition = self._print(expr.condition)
      body = self._print(expr.body)
      ret = f'if ({condition}) {{\n{body}\n}}'
      if expr.else_branch.args :
        else_branch = self._print(expr.else_branch)
        ret += f'\nelse {{\n{else_branch}\n}}'
      return ret
  def _print_Stream(self, strm):
    return self._print(strm.name)
  def _print_FPrint(self, expr) :
    stream = self._print(expr.file)
    fmt = self._print(expr.format_string)
    pargs = ', '.join(map(lambda arg: self._print(arg), expr.print_args))
    return f'fprintf({stream}, {fmt}' + (f', {pargs})' if pargs else ')')

def cxxcode(*args, **kw_args) :
  return _SpecificCodePrinter({'contract' : False, 'allow_unknown_functions' : True, 'order' : 'none'}).doprint(to_expr(args[0]), *args[1:], **kw_args)
#cxxcode = _SpecificCodePrinter({'contract' : False, 'allow_unknown_functions' : True, 'order' : 'none'}).doprint

class TmpVarManager :
  def __init__(self) :
    self.used_i_vars = []
    self.used_r_vars = []
    self.free_i_vars = []
    self.free_r_vars = []
  def get_int_var(self) :
    if self.free_i_vars :
      var = self.free_i_vars.pop()
    else :
      var = IDummy()
    self.used_i_vars.append(var)
    return var
  def get_real_var(self) :
    if self.free_r_vars :
      var = self.free_r_vars.pop()
    else :
      var = RDummy()
    self.used_r_vars.append(var)
    return var
  def rm_int_var(self, var) :
    try :
      self.used_i_vars.remove(var)
    except ValueError :
      print('TmpVarManager: Attempt to remove unused or non-existent integer var!')
      raise
  def rm_real_var(self, var) :
    try :
      self.used_r_vars.remove(var)
    except ValueError :
      print('TmpVarManager: Attempt to remove unused or non-existent real var!')
      raise
  def free_all_vars(self) :
    self.free_i_vars.extend(self.used_i_vars)
    self.free_r_vars.extend(self.used_r_vars)
    del self.used_i_vars[:]
    del self.used_r_vars[:]
  def get_declarations_and_clear(self) :
    res = cxxcode(CodeBlock(*(Declaration(Variable.deduced(var)) for var in self.used_i_vars),
                            *(Declaration(Variable.deduced(var)) for var in self.free_i_vars),
                            *(Declaration(Variable.deduced(var)) for var in self.used_r_vars),
                            *(Declaration(Variable.deduced(var)) for var in self.free_r_vars))).split('\n')
    del self.used_i_vars[:]
    del self.used_r_vars[:]
    del self.free_i_vars[:]
    del self.free_r_vars[:]
    return res

def decl_for_indices(indices) :
  if indices :
    return cxxcode(CodeBlock(*(Declaration(Variable.deduced(idx)) for idx in get_master_idx(indices) if idx.lower != idx.upper))).split('\n')
  else :
    return []

def elem_pos(strides, indices) :
  func = lambda expr, elem : expr * elem[0] + elem[1]
  return reduce(func, zip(strides, indices), S.Zero)

def indexed_for_array_elem(array_name, indices, offset, *, part_of = None, **assumptions) :
  part_of = part_of if part_of else indices
  shape = block_shape(part_of)
  offset = offset - elem_pos(shape, map(lambda idx : idx.lower, part_of))
  base = IndexedBaseWithOffset(array_name, shape = shape, offset = offset, **assumptions)
  return base[indices if indices else S.Zero]

def offset_in_block(indices, *, part_of = None) :
  part_of = part_of if part_of else indices
  strides = block_shape(part_of)
  return (elem_pos(strides, get_master_idx(indices))
          + elem_pos(strides, get_shifts(indices))
          - elem_pos(strides, map(lambda idx : idx.lower, part_of)))

def wrap_in_loop(body, indices) :
  func = lambda body, idx : [For(idx, Range(idx.lower, idx.upper + 1), body)] if len(idx) > 1 else body
  seq = reversed(get_master_idx(indices))
  return reduce(func, seq, body)

def assign_expr_with_sums(term, var_manager, replacements = [], force_sum_subs = []) :
  sum_subs = force_sum_subs if force_sum_subs else [(s, var_manager.get_real_var()) for s in get_sums(term)]
  term = term.subs(sum_subs)
  term_indices = set(get_outer_indices(term))
  preambula = []
  body = []
  blocks = []
  for s, d in sum_subs :
    s_term, *s_indices = s.args
    s_indices = tuple(idx for idx, _, _ in s_indices)
    new_s_indices = [idx.subs(idx.label, var_manager.get_int_var()) if len(idx) > 1 else idx for idx in s_indices]
    idx_subs = list(zip(s_indices, new_s_indices))
    s_term, s_term_indices, s_blocks, replacements = assign_expr_with_sums(s_term, var_manager, replacements)
    body = []
    for b in s_blocks :
      if b[0].isdisjoint(s_indices) :
        blocks.append(b)
      else :
        body.extend(code.subs(idx_subs) for code in b[1])
    # Здесь заменяем цикл умножением, если суммируется что-то, не зависящие от индексов суммирования
    if s_term_indices.isdisjoint(s_indices) :
      assert not body
      term = term.subs(d, block_size(s_indices) * s_term)
      var_manager.rm_real_var(d)
    else :
      s_term = s_term.subs(idx_subs)
      body.append(AddAugmentedAssignment(d, s_term))
      body = wrap_in_loop(body, new_s_indices)
      body.insert(0, Assignment(d, S(0.0)))
      s_term_indices.difference_update(s_indices)
      s_term_indices = {idx.subs(idx_subs) for idx in s_term_indices}
      blocks.append((s_term_indices, body))
    term_indices.update(s_term_indices)
  new_replacements = []
  for d, e in replacements :
    if term.has(d) :
      if isinstance(e, set) :
        term_indices.update(e)
        new_replacements.append((d,e))
      else :
        e_term, e_indices, e_blocks, new_replacements = assign_expr_with_sums(e, var_manager, new_replacements, [(e,d)] if isinstance(e, Sum) else [])
        blocks.extend(e_blocks)
        if not isinstance(e, Sum) :
          blocks.append((e_indices, [Assignment(d, e_term)]))
        term_indices.update(e_indices)
        new_replacements.append((d,e_indices))
    else :
      new_replacements.append((d,e))
  return term, term_indices, blocks, new_replacements

def subs_powers(expr, var_manager) :
  powers = []
  def do_subs(expr) :
    args = tuple(map(do_subs, expr.args))
    if isinstance(expr, Pow) :
      try :
        power = int(args[1])
      except TypeError :
        pass
      else :
        u_power = abs(power)
        if not args[0].is_Atom and power == args[1] and u_power > 1 :
          s = var_manager.get_real_var()
          powers.append((s, args[0]))
          return expr.func(s, *args[1:])
    return expr.func(*args) if args else expr
  expr = do_subs(expr)
  return powers, expr

def assign(array_names, values, master_indices, var_manager, *, pre_iter = [], post_iter = [], do_cse = True) :
  # Здесь предпологается, что все индексы суммирования уникальны. Это нужно только для использования cse().
  # Хорошо бы переопределить опеарцию сравнения Sum(), чтобы cse() мог выносить суммы, отличающиеся только символом индекса суммирования.
  new_master_indices = tuple(idx.subs(idx.label, var_manager.get_int_var()) if len(idx) > 1 else idx for idx in master_indices)
  idx_subs = list(zip(master_indices, new_master_indices))
  preambula = []
  body = [code.subs(idx_subs) for code in pre_iter]
  def rdummy_gen() :
    while True :
      yield var_manager.get_real_var()
  if do_cse :
    replacements, values = cse(values, rdummy_gen())
    # Так как Pow() с целым показателем будет превращен в произведение при кодогенерации, вынесем здесь основание степени как общее подвыражение
    new_replacements = []
    for rep in replacements :
      s, expr = subs_powers(rep[1], var_manager)
      new_replacements.extend(s)
      new_replacements.append((rep[0], expr))
    new_values = []
    for val in values :
      s, expr = subs_powers(val, var_manager)
      new_replacements.extend(s)
      new_values.append(expr)
    values = new_values
    replacements = new_replacements
  else :
    # не извлекаем общие подвыражения, оставляем это на компилятор (g++)
    replacements = []
  for name, val in zip(array_names, values) :
    term, term_indices, blocks, replacements = assign_expr_with_sums(val, var_manager, replacements)
    assert not (term_indices - set(master_indices))
    for b in blocks :
      if b[0].isdisjoint(master_indices) :
        preambula.extend(b[1])
      else :
        body.extend(code.subs(idx_subs) for code in b[1])
    body.extend([Assignment(name.subs(idx_subs), term.subs(idx_subs))])
  #assert not replacements
  body.extend(code.subs(idx_subs) for code in post_iter)
  body = wrap_in_loop(body, new_master_indices)
  if preambula :
    preambula.extend(body)
    body = preambula
  return cxxcode(CodeBlock(*body)).split('\n')

def assign_by_indices(array_names, values, indices, var_manager, offset, *, part_of = None) :
  name_to_code = lambda name : indexed_for_array_elem(name, indices, offset, part_of = part_of)
  array_names = map(name_to_code, array_names)
  return assign(array_names, values, get_master_idx(indices), var_manager)

def assign_by_counter(array_names, values, master_indices, var_manager, counter) :
  name_to_code = lambda name : Element(name, (counter,))
  array_names = map(name_to_code, array_names)
  return assign(array_names, values, master_indices, var_manager, post_iter = [PreIncrement(counter)])

if __name__ == "__main__" :
  from sympy import Symbol, Piecewise, sin, IndexedBase
  from sympy.codegen.ast import Stream, continue_
  from sympy2ipopt.utils.test_utils import renum_dummy
  from sympy2ipopt.idx_type import IdxType
  from sympy2ipopt.shifted_idx import ShiftedIdx
  from sympy2ipopt.utils.test_utils import check_value_error

  n = Symbol('n')
  x = Symbol('x')
  x_l = Symbol('x_l')
  x_u = Symbol('x_u')
  obj_value = Symbol('obj_value')
  counter = Symbol('counter')

  t1 = IdxType('t1', (0, 10))
  t2 = IdxType('t2', (2, 8))
  t3 = IdxType('t3', (-5, 3))
  t4 = IdxType('t4', (-9, -4))
  b1 = IndexedBaseWithOffset('b1', shape = (11,))
  b2 = IndexedBaseWithOffset('b2', shape = (7,))
  b3 = IndexedBaseWithOffset('b3', shape = (11, 7))

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

  sk3 = ShiftedIdx(k3, -1)

  r1 = IndexedBaseWithOffset('r', (1,), 3)
  r2 = IndexedBaseWithOffset('r', (1,), 5)
  q = IndexedBase('q', shape = (1, 1))

  assert cxxcode(x**3) == '(x*x*x)'
  assert cxxcode(x**-3) == '(1/(x*x*x))'
  assert cxxcode(empty_operator_) == ';'
  assert cxxcode(If(x > x_l, [Assignment(x, 5)])) == 'if (x > x_l) {\n   x = 5;\n}'
  assert cxxcode(If(x > x_l, [Assignment(x, 5)], [Assignment(x, 6)])) == 'if (x > x_l) {\n   x = 5;\n}\nelse {\n   x = 6;\n}'
  assert cxxcode(If(True, [Assignment(x, 5)], [Assignment(x, 6)])) == 'x = 5;'
  assert cxxcode(If(True, [], [Assignment(x, 6)])) == ';'
  assert cxxcode(If(False, [Assignment(x, 5)], [Assignment(x, 6)])) == 'x = 6;'
  assert cxxcode(If(False, [Assignment(x, 5)], [])) == ';'
  assert cxxcode(If(False, [Assignment(x, 5)])) == ';'

  assert cxxcode(Stream('my_file')) == 'my_file'
  assert cxxcode(FPrint((Symbol('x'),), '%f', 'stdout')) == 'fprintf(stdout, "%f", x)'
  assert cxxcode(FPrint((Symbol('x'), i1), '%f, %d\\n', 'stdout')) == 'fprintf(stdout, "%f, %d\\n", x, i1)'
  assert cxxcode(FPrint()) == 'fprintf(stdout, "")'

  assert elem_pos((11, 7, 9, 6), (i1, i2, i3, i4)) == 378 * i1 + 54 * i2 + 6 * i3 + i4

  assert offset_in_block((j1, sj2, i3, sj4)) == 81 * j1 + 27 * j2 + 3 * i3 + j4 - 112
  assert offset_in_block((j1, sj2, i3, sj4), part_of = (i1, i2, i3, i4)) == 378 * j1 + 54 * j2 + 6 * i3 + j4 - 175

  assert indexed_for_array_elem(x, (j1, sj2, i3, sj4), 5).base == IndexedBaseWithOffset(x, (10, 3, 9, 3), -55)
  assert cxxcode(indexed_for_array_elem(x, (j1, sj2, i3, sj4), 5)) == 'x[-107 + 3*i3 + 27*j2 + 81*j1 + j4]'
  assert indexed_for_array_elem(x, (k3,), 0, part_of = (i3,)).base == IndexedBaseWithOffset(x, (9,), 5)

  assert decl_for_indices((j1, sj2, i3, sj4)) == ['int j1;', 'int j2;', 'int i3;', 'int j4;']
  assert decl_for_indices((j1, k2, sk3, sj4)) == ['int j1;', 'int j4;']

  assert (cxxcode(CodeBlock(*wrap_in_loop([Assignment(indexed_for_array_elem(x, (j1, sj2, i3, sj4), 5), S(4))], (j1, sj2, i3, sj4)))).split('\n') ==
          ['for (j1 = 0; j1 < 10; j1 += 1) {',
           '   for (j2 = 5; j2 < 8; j2 += 1) {',
           '      for (i3 = -5; i3 < 4; i3 += 1) {',
           '         for (j4 = -8; j4 < -5; j4 += 1) {',
           '            x[-107 + 3*i3 + 27*j2 + 81*j1 + j4] = 4;',
           '         };',
           '      };',
           '   };',
           '};'])

  assert (cxxcode(CodeBlock(*wrap_in_loop([Assignment(indexed_for_array_elem(x, (j1, k2, sk3, sj4), 5), S(4))], (j1, k2, sk3, sj4)))).split('\n') ==
          ['for (j1 = 0; j1 < 10; j1 += 1) {',
           '   for (j4 = -8; j4 < -5; j4 += 1) {',
           '      x[13 + 3*j1 + j4] = 4;',
           '   };',
           '};'])

  assert cxxcode(CodeBlock(*wrap_in_loop([Assignment(indexed_for_array_elem(x, (), 5), S(4))], ()))).split('\n') == ['x[5] = 4;']


  var_manager = TmpVarManager()
  tmp1 = var_manager.get_int_var()
  tmp2 = var_manager.get_real_var()
  var_manager.rm_int_var(tmp1)
  check_value_error(var_manager.rm_int_var, tmp1)
  check_value_error(var_manager.rm_int_var, tmp2)
  var_manager.rm_real_var(tmp2)
  assert var_manager.get_declarations_and_clear() == ['']
  tmp1 = var_manager.get_int_var()
  tmp2 = var_manager.get_real_var()
  var_manager.free_all_vars()
  tmp1 = var_manager.get_int_var()
  tmp2 = var_manager.get_real_var()
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_1;', 'double _Dummy_2;']

  t, i, b, r = assign_expr_with_sums(Sum(r1[i1] + r2[i2], i2), var_manager)
  assert i == {i1}
  assert len(b) == 1
  assert b[0][0] == {i1}
  assert not r
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_2;', 'double _Dummy_1;']
  assert renum_dummy(cxxcode(CodeBlock(*b[0][1])).split('\n')) == ['_Dummy_1 = 0.0;',
                                                             'for (_Dummy_2 = 2; _Dummy_2 < 9; _Dummy_2 += 1) {',
                                                             '   _Dummy_1 += r[3 + i1] + r[5 + _Dummy_2];',
                                                             '};']
  assert renum_dummy(cxxcode(t)) == '_Dummy_1'

  t, i, b, r = assign_expr_with_sums(Sum(r1[i1] + r2[i2], i2), var_manager)
  assert i == {i1}
  assert len(b) == 1
  assert not r
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_2;', 'double _Dummy_1;']
  assert b[0][0] == {i1}
  assert renum_dummy(cxxcode(CodeBlock(*b[0][1])).split('\n')) == ['_Dummy_1 = 0.0;',
                                                                   'for (_Dummy_2 = 2; _Dummy_2 < 9; _Dummy_2 += 1) {',
                                                                   '   _Dummy_1 += r[3 + i1] + r[5 + _Dummy_2];',
                                                                   '};']
  assert renum_dummy(cxxcode(t)) == '_Dummy_1'

  t, i, b, r = assign_expr_with_sums(Sum(r1[i1] + Sum(r2[i2], i2), i1), var_manager)
  assert not i
  assert len(b) == 2
  assert not b[0][0]
  assert not b[1][0]
  assert not r
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_2;',
                                                                   'int _Dummy_4;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_3;']
  assert renum_dummy(cxxcode(CodeBlock(*b[0][1], *b[1][1])).split('\n')) == ['_Dummy_3 = 0.0;',
                                                                             'for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                                             '   _Dummy_3 += r[5 + _Dummy_4];',
                                                                             '};',
                                                                             '_Dummy_1 = 0.0;',
                                                                             'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                                             '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                                             '};']
  assert renum_dummy(cxxcode(t)) == '_Dummy_1'

  t, i, b, r = assign_expr_with_sums(Sum(r1[i1] + Sum(r2[i2] + Sum(q[i1, i3], i3), i2), i1), var_manager)
  assert not i
  assert len(b) == 1
  assert not b[0][0]
  assert not r
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_2;',
                                                                   'int _Dummy_4;',
                                                                   'int _Dummy_6;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_3;',
                                                                   'double _Dummy_5;']
  assert renum_dummy(cxxcode(CodeBlock(*b[0][1])).split('\n')) == ['_Dummy_1 = 0.0;',
                                                                   'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                                   '   _Dummy_5 = 0.0;',
                                                                   '   for (_Dummy_6 = -5; _Dummy_6 < 4; _Dummy_6 += 1) {',
                                                                   '      _Dummy_5 += q[_Dummy_2 + _Dummy_6];',
                                                                   '   };',
                                                                   '   _Dummy_3 = 0.0;',
                                                                   '   for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                                   '      _Dummy_3 += _Dummy_5 + r[5 + _Dummy_4];',
                                                                   '   };',
                                                                   '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                                   '};']
  assert renum_dummy(cxxcode(t)) == '_Dummy_1'

  t, i, b, r = assign_expr_with_sums(Sum(r1[i1] + Sum(r2[i2] + Sum(q[i1, i3], i1, i3), i2), i1), var_manager)
  assert not i
  assert len(b) == 3
  assert not b[0][0]
  assert not b[1][0]
  assert not b[2][0]
  assert not r
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_2;',
                                                                   'int _Dummy_4;',
                                                                   'int _Dummy_6;',
                                                                   'int _Dummy_7;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_3;',
                                                                   'double _Dummy_5;']
  assert renum_dummy(cxxcode(CodeBlock(*b[0][1], *b[1][1], *b[2][1])).split('\n')) == ['_Dummy_5 = 0.0;',
                                                                                       'for (_Dummy_6 = 0; _Dummy_6 < 11; _Dummy_6 += 1) {',
                                                                                       '   for (_Dummy_7 = -5; _Dummy_7 < 4; _Dummy_7 += 1) {',
                                                                                       '      _Dummy_5 += q[_Dummy_6 + _Dummy_7];',
                                                                                       '   };',
                                                                                       '};',
                                                                                       '_Dummy_3 = 0.0;',
                                                                                       'for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                                                       '   _Dummy_3 += _Dummy_5 + r[5 + _Dummy_4];',
                                                                                       '};',
                                                                                       '_Dummy_1 = 0.0;',
                                                                                       'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                                                       '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                                                       '};']
  assert renum_dummy(cxxcode(t)) == '_Dummy_1'

  t, i, b, r = assign_expr_with_sums(Sum(r1[i1] + Sum(r2[i2] + Sum(q[i2, i3], i3), i2), i1), var_manager)
  assert not i
  assert len(b) == 2
  assert not b[0][0]
  assert not b[1][0]
  assert not r
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_2;',
                                                                   'int _Dummy_4;',
                                                                   'int _Dummy_6;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_3;',
                                                                   'double _Dummy_5;']
  assert renum_dummy(cxxcode(CodeBlock(*b[0][1], *b[1][1])).split('\n')) == ['_Dummy_3 = 0.0;',
                                                                             'for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                                             '   _Dummy_5 = 0.0;',
                                                                             '   for (_Dummy_6 = -5; _Dummy_6 < 4; _Dummy_6 += 1) {',
                                                                             '      _Dummy_5 += q[_Dummy_4 + _Dummy_6];',
                                                                             '   };',
                                                                             '   _Dummy_3 += _Dummy_5 + r[5 + _Dummy_4];',
                                                                             '};',
                                                                             '_Dummy_1 = 0.0;',
                                                                             'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                                             '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                                             '};']
  assert renum_dummy(cxxcode(t)) == '_Dummy_1'

  t, i, b, r = assign_expr_with_sums(Sum(r1[i1] + Sum(r2[i2] + Sum(q[i1, i2], i3), i2), i1), var_manager)
  assert not i
  assert len(b) == 1
  assert not b[0][0]
  assert not r
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_2;',
                                                                   'int _Dummy_4;',
                                                                   'int _Dummy_6;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_3;']
  assert renum_dummy(cxxcode(CodeBlock(*b[0][1])).split('\n')) == ['_Dummy_1 = 0.0;',
                                                                   'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                                   '   _Dummy_3 = 0.0;',
                                                                   '   for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                                   '      _Dummy_3 += 9*q[_Dummy_2 + _Dummy_4] + r[5 + _Dummy_4];',
                                                                   '   };',
                                                                   '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                                   '};']
  assert renum_dummy(cxxcode(t)) == '_Dummy_1'

  t, i, b, r = assign_expr_with_sums(Sum(r1[i1] + Sum(r2[i3] + Sum(q[i1, i1] + Sum(q[i1, i2], i2), i3), i2), i1), var_manager)
  assert i == {i3}
  assert len(b) == 1
  assert b[0][0] == {i3}
  assert not r
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_2;',
                                                                   'int _Dummy_4;',
                                                                   'int _Dummy_6;',
                                                                   'int _Dummy_8;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_7;']
  assert renum_dummy(cxxcode(CodeBlock(*b[0][1])).split('\n')) == ['_Dummy_1 = 0.0;',
                                                                   'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                                   '   _Dummy_7 = 0.0;',
                                                                   '   for (_Dummy_8 = 2; _Dummy_8 < 9; _Dummy_8 += 1) {',
                                                                   '      _Dummy_7 += q[_Dummy_2 + _Dummy_8];',
                                                                   '   };',
                                                                   '   _Dummy_1 += 7*r[5 + i3] + 63*_Dummy_7 + 63*q[2*_Dummy_2] + r[3 + _Dummy_2];',
                                                                   '};']
  assert renum_dummy(cxxcode(t)) == '_Dummy_1'

  s, expr = subs_powers((x_u+(r2[i2]+x_l)**4)**2-r1[i1]**x, var_manager)
  assert len (s) == 2
  assert renum_dummy(cxxcode(expr)) == '(_Dummy_1*_Dummy_1) - std::pow(r[3 + i1], x)'
  assert renum_dummy(cxxcode(s[0][1])) == 'x_l + r[5 + i2]'
  assert renum_dummy(cxxcode(s[1][1])) == 'x_u + (_Dummy_1*_Dummy_1*_Dummy_1*_Dummy_1)'
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['double _Dummy_1;', 'double _Dummy_2;']

  assert assign((n, obj_value), (S(100), x), (), var_manager) == ['n = 100;', 'obj_value = x;']
  assert var_manager.get_declarations_and_clear() == ['']

  assert (renum_dummy(assign((IndexedBase(x)[i1], IndexedBase(x_l)[i1], IndexedBase(x_u)[i1]),
                       (b1[i1]**(b3[i1, i2] + b1[i1]), Sum(b1[i1], i1) + b3[i1, i2], Sum(b3[i1, i2], i1) + Sum(b2[j2], j2) + b3[i1, i2]),
                       (i1, i2),
                       var_manager,
                       pre_iter = [If(i1 > i2, [continue_])],
                       post_iter = [PreIncrement(counter)],
                       do_cse = True
                      )) == 
          ['_Dummy_3 = 0.0;',
           'for (_Dummy_4 = 0; _Dummy_4 < 11; _Dummy_4 += 1) {',
           '   _Dummy_3 += b1[_Dummy_4];',
           '};',
           '_Dummy_5 = 0.0;',
           'for (_Dummy_7 = 5; _Dummy_7 < 8; _Dummy_7 += 1) {',
           '   _Dummy_5 += b2[_Dummy_7];',
           '};',
           'for (_Dummy_1 = 0; _Dummy_1 < 11; _Dummy_1 += 1) {',
           '   for (_Dummy_2 = 2; _Dummy_2 < 9; _Dummy_2 += 1) {',
           '      if (_Dummy_1 > _Dummy_2) {',
           '         continue;',
           '      };',
           '      x[_Dummy_1] = std::pow(b1[_Dummy_1], b1[_Dummy_1] + b3[7*_Dummy_1 + _Dummy_2]);',
           '      x_l[_Dummy_1] = _Dummy_3 + b3[7*_Dummy_1 + _Dummy_2];',
           '      _Dummy_6 = 0.0;',
           '      for (_Dummy_8 = 0; _Dummy_8 < 11; _Dummy_8 += 1) {',
           '         _Dummy_6 += b3[7*_Dummy_8 + _Dummy_2];',
           '      };',
           '      x_u[_Dummy_1] = _Dummy_5 + _Dummy_6 + b3[7*_Dummy_1 + _Dummy_2];',
           '      ++(counter);',
           '   };',
           '};'])
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_1;',
                                                                   'int _Dummy_2;',
                                                                   'int _Dummy_4;',
                                                                   'int _Dummy_7;',
                                                                   'int _Dummy_8;',
                                                                   'double _Dummy_3;',
                                                                   'double _Dummy_5;',
                                                                   'double _Dummy_6;']

  b4 = IndexedBaseWithOffset('b4', shape = (10, 3, 1))
  b5 = IndexedBaseWithOffset('b5', shape = (3, 9, 3))
  assert (renum_dummy(assign_by_indices((x_l, x_u), (b4[j1, sj2, sk3], b5[sj2, k3, sj4]), (j1, sj2, sk3, sj4), var_manager, 5)) ==
          ['for (_Dummy_1 = 0; _Dummy_1 < 10; _Dummy_1 += 1) {',
           '   for (_Dummy_2 = 5; _Dummy_2 < 8; _Dummy_2 += 1) {',
           '      for (_Dummy_3 = -8; _Dummy_3 < -5; _Dummy_3 += 1) {',
           '         x_l[-2 + 3*_Dummy_2 + 9*_Dummy_1 + _Dummy_3] = b4[-3 + 3*_Dummy_1 + _Dummy_2];',
           '         x_u[-2 + 3*_Dummy_2 + 9*_Dummy_1 + _Dummy_3] = b5[-52 + 27*_Dummy_2 + _Dummy_3];',
           '      };',
           '   };',
           '};'])
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_1;',
                                                                   'int _Dummy_2;',
                                                                   'int _Dummy_3;']


  assert (renum_dummy(assign_by_counter((x_l, x_u), (b4[j1, sj2, sk3], b5[sj2, k3, sj4]), (j1, j2, k3, j4), var_manager, counter)) ==
          ['for (_Dummy_1 = 0; _Dummy_1 < 10; _Dummy_1 += 1) {',
          '   for (_Dummy_2 = 5; _Dummy_2 < 8; _Dummy_2 += 1) {',
          '      for (_Dummy_3 = -8; _Dummy_3 < -5; _Dummy_3 += 1) {',
          '         x_l[counter] = b4[-3 + 3*_Dummy_1 + _Dummy_2];',
          '         x_u[counter] = b5[-52 + 27*_Dummy_2 + _Dummy_3];',
          '         ++(counter);',
          '      };',
          '   };',
          '};'])
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_1;',
                                                                   'int _Dummy_2;',
                                                                   'int _Dummy_3;']

  assert renum_dummy(assign((obj_value,), (sin(x)**2 / sin(x)**sin(x),), (), var_manager, do_cse = True)) == [
                                                                                               '_Dummy_1 = std::sin(x);',
                                                                                               'obj_value = (_Dummy_1*_Dummy_1)*std::pow(_Dummy_1, -_Dummy_1);']
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['double _Dummy_1;']

  assert renum_dummy(assign((obj_value, Symbol('y')), ((sin(x)+1)**2 / sin(x)**(sin(x)+1), sin(x)**3+sin(x)+1), (), var_manager, do_cse = True)) == [
                                                                                             '_Dummy_1 = std::sin(x);',
                                                                                             '_Dummy_2 = 1 + _Dummy_1;',
                                                                                             'obj_value = std::pow(_Dummy_1, -1 - _Dummy_1)*(_Dummy_2*_Dummy_2);',
                                                                                             'y = (_Dummy_1*_Dummy_1*_Dummy_1) + _Dummy_2;']
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['double _Dummy_1;',
                                                                   'double _Dummy_2;']

  assert renum_dummy(assign((obj_value, Symbol('y')), (Sum(b1[i1]*(sin(x)+1)**2 / sin(x)**(sin(x)+1), i1), Sum(sin(x)**3+Sum(b3[i1,i2]*(sin(x)+1), i1), i2)),
                            (), var_manager, do_cse = True)) == [
                                                  '_Dummy_1 = std::sin(x);',
                                                  '_Dummy_2 = 1 + _Dummy_1;',
                                                  '_Dummy_3 = 0.0;',
                                                  'for (_Dummy_4 = 0; _Dummy_4 < 11; _Dummy_4 += 1) {',
                                                  '   _Dummy_3 += std::pow(_Dummy_1, -1 - _Dummy_1)*(_Dummy_2*_Dummy_2)*b1[_Dummy_4];',
                                                  '};',
                                                  '_Dummy_5 = 0.0;',
                                                  'for (_Dummy_6 = 2; _Dummy_6 < 9; _Dummy_6 += 1) {',
                                                  '   _Dummy_7 = 0.0;',
                                                  '   for (_Dummy_8 = 0; _Dummy_8 < 11; _Dummy_8 += 1) {',
                                                  '      _Dummy_7 += _Dummy_2*b3[7*_Dummy_8 + _Dummy_6];',
                                                  '   };',
                                                  '   _Dummy_5 += (_Dummy_1*_Dummy_1*_Dummy_1) + _Dummy_7;', '};',
                                                  'obj_value = _Dummy_3;',
                                                  'y = _Dummy_5;']
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_4;',
                                                                   'int _Dummy_6;',
                                                                   'int _Dummy_8;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_2;',
                                                                   'double _Dummy_3;',
                                                                   'double _Dummy_5;',
                                                                   'double _Dummy_7;']

  assert renum_dummy(assign((obj_value, Symbol('y')),
                            (Sum(b1[i1]*(sin(x)+1)**2 / sin(x)**(sin(x)+1), i1), Sum((b2[i2] + sin(x)**3)/Sum(b3[i1,i2]*(sin(x)+1 + b2[i2] +sin(x)**3), i1), i2)),
                            (), var_manager, do_cse = True)) == [
                                                  '_Dummy_1 = std::sin(x);',
                                                  '_Dummy_2 = 1 + _Dummy_1;',
                                                  '_Dummy_4 = 0.0;',
                                                  'for (_Dummy_5 = 0; _Dummy_5 < 11; _Dummy_5 += 1) {',
                                                  '   _Dummy_4 += std::pow(_Dummy_1, -1 - _Dummy_1)*(_Dummy_2*_Dummy_2)*b1[_Dummy_5];',
                                                  '};',
                                                  '_Dummy_6 = 0.0;',
                                                  'for (_Dummy_7 = 2; _Dummy_7 < 9; _Dummy_7 += 1) {',
                                                  '   _Dummy_3 = (_Dummy_1*_Dummy_1*_Dummy_1) + b2[_Dummy_7];',
                                                  '   _Dummy_8 = 0.0;',
                                                  '   for (_Dummy_9 = 0; _Dummy_9 < 11; _Dummy_9 += 1) {',
                                                  '      _Dummy_8 += (_Dummy_2 + _Dummy_3)*b3[7*_Dummy_9 + _Dummy_7];',
                                                  '   };',
                                                  '   _Dummy_6 += _Dummy_3/_Dummy_8;',
                                                  '};',
                                                  'obj_value = _Dummy_4;',
                                                  'y = _Dummy_6;']

  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_5;',
                                                                   'int _Dummy_7;',
                                                                   'int _Dummy_9;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_2;',
                                                                   'double _Dummy_3;',
                                                                   'double _Dummy_4;',
                                                                   'double _Dummy_6;',
                                                                   'double _Dummy_8;']

  assert renum_dummy(assign((obj_value, Symbol('y')), (Sum(Sum(b1[i1]*(sin(x)+1)**2, i1) + b2[i2], i2), Sum(b1[i1]*(sin(x)+1)**2, i1)/(sin(x) + 1)),
                            (), var_manager,do_cse = True)) == [
                                                  '_Dummy_1 = 1 + std::sin(x);',
                                                  '_Dummy_2 = 0.0;',
                                                  'for (_Dummy_5 = 0; _Dummy_5 < 11; _Dummy_5 += 1) {',
                                                  '   _Dummy_2 += (_Dummy_1*_Dummy_1)*b1[_Dummy_5];',
                                                  '};',
                                                  '_Dummy_3 = 0.0;',
                                                  'for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                  '   _Dummy_3 += _Dummy_2 + b2[_Dummy_4];',
                                                  '};',
                                                  'obj_value = _Dummy_3;',
                                                  'y = _Dummy_2/_Dummy_1;']
  assert renum_dummy(var_manager.get_declarations_and_clear()) == ['int _Dummy_4;',
                                                                   'int _Dummy_5;',
                                                                   'double _Dummy_1;',
                                                                   'double _Dummy_2;',
                                                                   'double _Dummy_3;']

  print('ALL TESTS HAVE BEEN PASSED!!!')
