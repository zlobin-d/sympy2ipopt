#!/bin/python3

from sympy import Symbol, IndexedBase, Idx, S, Sum, Range
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
  ''' CXX11CodePrinter в который добавлена поддержка пустого оператора EmptyOperator и условного оператора If. '''
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
  func = lambda body, idx : [For(idx, Range(idx.lower, idx.upper + 1), body)] if idx.lower != idx.upper else body
  seq = reversed(get_master_idx(indices))
  return reduce(func, seq, body)

def assign_one(name, val, indices = (), op = Assignment) :
  indices = set(get_master_idx(indices))
  preambula = []
  def process_sum(expr) :
    bodies = []
    sum_subs = [(s, RDummy()) for s in get_sums(expr)]
    expr = expr.subs(sum_subs)
    expr_indices = set(get_outer_indices(expr))
    #expr_indices = set(sym for sym in expr.free_symbols if isinstance(sym, Idx))
    for s, d in sum_subs :
      term, *s_indices = s.args
      s_indices = [idx for idx, _, _ in s_indices]
      new_s_indices = [idx.subs(idx.label, IDummy()) for idx in s_indices]
      preambula.extend(Variable.deduced(new_idx).as_Declaration() for new_idx in new_s_indices)
      term, term_indices, term_bodies = process_sum(term.subs(zip(s_indices, new_s_indices)))
      expr_indices.update(term_indices)
      s_body = []
      for elem in term_bodies :
        if elem[0].isdisjoint(new_s_indices) :
          bodies.append(elem)
        else :
          preambula.append(Variable.deduced(elem[1]).as_Declaration())
          s_body.append(Assignment(elem[1], S(0.0)))
          s_body.extend(elem[2])
      if term_indices.isdisjoint(new_s_indices) :
        assert not s_body
        expr = expr.subs(d, block_size(new_s_indices) * term)
      else :
        s_body.append(AddAugmentedAssignment(d, term))
        bodies.append((term_indices, d, wrap_in_loop(s_body, new_s_indices)))
        expr_indices.difference_update(new_s_indices)
    return expr, expr_indices, bodies
  val, val_indices, bodies = process_sum(val)
  body = []
  for elem in bodies :
    if elem[0].isdisjoint(indices) :
      preambula.append(Variable.deduced(elem[1], value = S(0.0)).as_Declaration())
      preambula.extend(elem[2])
    else :
      preambula.append(Variable.deduced(elem[1]).as_Declaration())
      body.append(Assignment(elem[1], S(0.0)))
      body.extend(elem[2])
  body.append(op(name, val))
  return preambula, body

def assign(array_names, values, indices = (), name_to_code = lambda name : name, on_every_iter = None) :
  preambula = []
  body = []
  for name, val in zip(array_names, values) :
    p, b = assign_one(name_to_code(name), val, indices)
    preambula.extend(p)
    body.extend(b)
  if on_every_iter :
    body.extend(on_every_iter)
  body = wrap_in_loop(body, indices)
  if preambula :
    preambula.extend(body)
    body = preambula
  return cxxcode(CodeBlock(*body)).split('\n')

def assign_by_indices(array_names, values, indices, offset, *, part_of = None) :
  name_to_code = lambda name : indexed_for_array_elem(name, indices, offset, part_of = part_of)
  return assign(array_names, values, indices, name_to_code)

def assign_by_counter(array_names, values, indices, counter) :
  name_to_code = lambda name : Element(name, (counter,))
  return assign(array_names, values, indices, name_to_code, [PreIncrement(counter)])

if __name__ == "__main__" :
  from sympy import Symbol, Piecewise
  from sympy.codegen.ast import Stream
  from sympy2ipopt.utils.test_utils import renum_dummy
  from sympy2ipopt.idx_type import IdxType
  from sympy2ipopt.shifted_idx import ShiftedIdx

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


  p, b = assign_one(b1[i1], Sum(r1[i1] + r2[i2], i2), (i1,), op = AddAugmentedAssignment)
  assert renum_dummy(cxxcode(CodeBlock(*p)).split('\n')) == ['int _Dummy_2;', 'double _Dummy_1;']
  assert renum_dummy(cxxcode(CodeBlock(*b)).split('\n')) == ['_Dummy_1 = 0.0;',
                                                             'for (_Dummy_2 = 2; _Dummy_2 < 9; _Dummy_2 += 1) {',
                                                             '   _Dummy_1 += r[3 + i1] + r[5 + _Dummy_2];',
                                                             '};',
                                                             'b1[i1] += _Dummy_1;']
  p, b = assign_one(b2[i2], Sum(r1[i1] + r2[i2], i2), (i2,))
  assert renum_dummy(cxxcode(CodeBlock(*p)).split('\n')) == ['int _Dummy_2;',
                                                             'double _Dummy_1 = 0.0;',
                                                             'for (_Dummy_2 = 2; _Dummy_2 < 9; _Dummy_2 += 1) {',
                                                             '   _Dummy_1 += r[3 + i1] + r[5 + _Dummy_2];',
                                                             '};']
  assert renum_dummy(cxxcode(CodeBlock(*b)).split('\n')) == ['b2[i2] = _Dummy_1;']

  p, b = assign_one(b2[i2], Sum(r1[i1] + Sum(r2[i2], i2), i1), (i2,))
  assert renum_dummy(cxxcode(CodeBlock(*p)).split('\n')) == ['int _Dummy_2;',
                                                             'int _Dummy_4;',
                                                             'double _Dummy_3 = 0.0;',
                                                             'for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                             '   _Dummy_3 += r[5 + _Dummy_4];',
                                                             '};',
                                                             'double _Dummy_1 = 0.0;',
                                                             'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                             '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                             '};']
  assert renum_dummy(cxxcode(CodeBlock(*b)).split('\n')) == ['b2[i2] = _Dummy_1;']

  p, b = assign_one(b2[i2], Sum(r1[i1] + Sum(r2[i2] + Sum(q[i1, i3], i3), i2), i1), (i2,))
  assert renum_dummy(cxxcode(CodeBlock(*p)).split('\n')) == ['int _Dummy_2;',
                                                             'int _Dummy_4;',
                                                             'int _Dummy_6;',
                                                             'double _Dummy_5;',
                                                             'double _Dummy_3;',
                                                             'double _Dummy_1 = 0.0;',
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
  assert renum_dummy(cxxcode(CodeBlock(*b)).split('\n')) == ['b2[i2] = _Dummy_1;']

  p, b = assign_one(b2[i2], Sum(r1[i1] + Sum(r2[i2] + Sum(q[i1, i3], i1, i3), i2), i1), (i2,))
  assert renum_dummy(cxxcode(CodeBlock(*p)).split('\n')) == ['int _Dummy_2;',
                                                             'int _Dummy_4;',
                                                             'int _Dummy_6;',
                                                             'int _Dummy_7;', 'double _Dummy_5 = 0.0;',
                                                             'for (_Dummy_6 = 0; _Dummy_6 < 11; _Dummy_6 += 1) {',
                                                             '   for (_Dummy_7 = -5; _Dummy_7 < 4; _Dummy_7 += 1) {',
                                                             '      _Dummy_5 += q[_Dummy_6 + _Dummy_7];',
                                                             '   };',
                                                             '};',
                                                             'double _Dummy_3 = 0.0;',
                                                             'for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                             '   _Dummy_3 += _Dummy_5 + r[5 + _Dummy_4];',
                                                             '};',
                                                             'double _Dummy_1 = 0.0;',
                                                             'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                             '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                             '};']
  assert renum_dummy(cxxcode(CodeBlock(*b)).split('\n')) == ['b2[i2] = _Dummy_1;']

  p, b = assign_one(b2[i2], Sum(r1[i1] + Sum(r2[i2] + Sum(q[i2, i3], i3), i2), i1), (i2,))
  assert renum_dummy(cxxcode(CodeBlock(*p)).split('\n')) == ['int _Dummy_2;',
                                                             'int _Dummy_4;',
                                                             'int _Dummy_6;',
                                                             'double _Dummy_5;',
                                                             'double _Dummy_3 = 0.0;',
                                                             'for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                             '   _Dummy_5 = 0.0;',
                                                             '   for (_Dummy_6 = -5; _Dummy_6 < 4; _Dummy_6 += 1) {',
                                                             '      _Dummy_5 += q[_Dummy_4 + _Dummy_6];',
                                                             '   };',
                                                             '   _Dummy_3 += _Dummy_5 + r[5 + _Dummy_4];',
                                                             '};',
                                                             'double _Dummy_1 = 0.0;',
                                                             'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                             '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                             '};']
  assert renum_dummy(cxxcode(CodeBlock(*b)).split('\n')) == ['b2[i2] = _Dummy_1;']

  p, b = assign_one(b2[i2], Sum(r1[i1] + Sum(r2[i2] + Sum(q[i1, i2], i3), i2), i1), (i2,))
  assert renum_dummy(cxxcode(CodeBlock(*p)).split('\n')) == ['int _Dummy_2;',
                                                             'int _Dummy_4;',
                                                             'int _Dummy_6;',
                                                             'double _Dummy_3;',
                                                             'double _Dummy_1 = 0.0;',
                                                             'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                             '   _Dummy_3 = 0.0;',
                                                             '   for (_Dummy_4 = 2; _Dummy_4 < 9; _Dummy_4 += 1) {',
                                                             '      _Dummy_3 += 9*q[_Dummy_2 + _Dummy_4] + r[5 + _Dummy_4];',
                                                             '   };',
                                                             '   _Dummy_1 += _Dummy_3 + r[3 + _Dummy_2];',
                                                             '};']
  assert renum_dummy(cxxcode(CodeBlock(*b)).split('\n')) == ['b2[i2] = _Dummy_1;']

  p, b = assign_one(b2[i2], Sum(r1[i1] + Sum(r2[i3] + Sum(q[i1, i1] + Sum(q[i1, i2], i2), i3), i2), i1), (i2,))
  assert renum_dummy(cxxcode(CodeBlock(*p)).split('\n')) == ['int _Dummy_2;',
                                                             'int _Dummy_4;',
                                                             'int _Dummy_6;',
                                                             'int _Dummy_8;',
                                                             'double _Dummy_7;',
                                                             'double _Dummy_1 = 0.0;',
                                                             'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
                                                             '   _Dummy_7 = 0.0;',
                                                             '   for (_Dummy_8 = 2; _Dummy_8 < 9; _Dummy_8 += 1) {',
                                                             '      _Dummy_7 += q[_Dummy_2 + _Dummy_8];',
                                                             '   };',
                                                             '   _Dummy_1 += 7*r[5 + i3] + 63*_Dummy_7 + 63*q[2*_Dummy_2] + r[3 + _Dummy_2];',
                                                             '};']
  assert renum_dummy(cxxcode(CodeBlock(*b)).split('\n')) == ['b2[i2] = _Dummy_1;']

  assert assign((n, obj_value), (S(100), r1[i1] + r2[i1])) == ['n = 100;', 'obj_value = r[3 + i1] + r[5 + i1];']

  assert (renum_dummy(assign((x, x_l, x_u),
                       (b1[i1]**(b3[i1, i2] + b1[i1]), Sum(b1[i1], i1) + b3[i1, i2], Sum(b3[i1, i2], i1) + Sum(b2[j2], j2) + b3[i1, i2]),
                       (i1, i2),
                       lambda name : IndexedBase(name)[i1],
                       [PreIncrement(counter)]
                      )) == 
          ['int _Dummy_2;',
           'double _Dummy_1 = 0.0;',
           'for (_Dummy_2 = 0; _Dummy_2 < 11; _Dummy_2 += 1) {',
           '   _Dummy_1 += b1[_Dummy_2];',
           '};',
           'int _Dummy_5;',
           'int _Dummy_6;',
           'double _Dummy_3 = 0.0;',
           'for (_Dummy_5 = 5; _Dummy_5 < 8; _Dummy_5 += 1) {',
           '   _Dummy_3 += b2[_Dummy_5];',
           '};',
           'double _Dummy_4;',
           'for (i1 = 0; i1 < 11; i1 += 1) {',
           '   for (i2 = 2; i2 < 9; i2 += 1) {',
           '      x[i1] = std::pow(b1[i1], b1[i1] + b3[7*i1 + i2]);',
           '      x_l[i1] = _Dummy_1 + b3[7*i1 + i2];',
           '      _Dummy_4 = 0.0;',
           '      for (_Dummy_6 = 0; _Dummy_6 < 11; _Dummy_6 += 1) {',
           '         _Dummy_4 += b3[7*_Dummy_6 + i2];',
           '      };',
           '      x_u[i1] = _Dummy_3 + _Dummy_4 + b3[7*i1 + i2];',
           '      ++(counter);',
           '   };',
           '};'])

  b4 = IndexedBaseWithOffset('b4', shape = (10, 3, 1))
  b5 = IndexedBaseWithOffset('b5', shape = (3, 9, 3))
  assert (assign_by_indices((x_l, x_u), (b4[j1, sj2, sk3], b5[sj2, i3, sj4]), (j1, sj2, sk3, sj4), 5) ==
          ['for (j1 = 0; j1 < 10; j1 += 1) {',
           '   for (j2 = 5; j2 < 8; j2 += 1) {',
           '      for (j4 = -8; j4 < -5; j4 += 1) {',
           '         x_l[-2 + 3*j2 + 9*j1 + j4] = b4[-3 + 3*j1 + j2];',
           '         x_u[-2 + 3*j2 + 9*j1 + j4] = b5[-52 + 3*i3 + 27*j2 + j4];',
           '      };',
           '   };',
           '};'])

  assert (assign_by_counter((x_l, x_u), (b4[j1, sj2, sk3], b5[sj2, i3, sj4]), (j1, sj2, sk3, sj4), counter) ==
          ['for (j1 = 0; j1 < 10; j1 += 1) {',
           '   for (j2 = 5; j2 < 8; j2 += 1) {',
           '      for (j4 = -8; j4 < -5; j4 += 1) {',
           '         x_l[counter] = b4[-3 + 3*j1 + j2];',
           '         x_u[counter] = b5[-52 + 3*i3 + 27*j2 + j4];',
           '         ++(counter);',
           '      };',
           '   };',
           '};'])

  print('ALL TESTS HAVE BEEN PASSED!!!')
