#!/bin/python3

from collections import defaultdict
from functools import reduce
from sympy import Eq, srepr
from sympy.codegen.ast import CodeBlock, continue_
from sympy2ipopt.idx_type import IdxOutOfRangeError
from sympy2ipopt.shifted_idx import ShiftedIdx
from sympy2ipopt.utils.idx_utils import get_master_idx, get_shifts, get_types, IDummy, idx_subs, block_copy, block_size
from sympy2ipopt.utils.code_utils import If, cxxcode, wrap_in_loop

def _master_block_size(indices) :
  return block_size(set(get_master_idx(indices)))

# Как идея, дробить все на самые мелкие кусочки (декартово произведение), а склеивать потом, после отбрасывания наддиагональных элементов гессиана.
# Это позволит иметь полностью симметричную картину частей для диагональных блоков гессиана: из двух симметричных блоков можно один выбросить, а другой отразить при помощи if.
def _intersect_idx(idx1, idx2) :
  ''' Находим пересечение двух диапазонов. Результат -- общий диапазон,
      и, возможно пустые, списки диапазонов, принадлижащих только одному из исходных диапазонов. '''
  idx_types = get_types((idx1, idx2))
  assert idx_types[0] == idx_types[1]
  new_idx = lambda l, u : idx_types[0](IDummy(), (l,u))
  if idx1.upper < idx2.lower or idx2.upper < idx1.lower :
    return None, [idx1], [idx2]
  elif idx1.lower < idx2.lower :
    if idx1.upper < idx2.upper :
      return (new_idx(idx2.lower, idx1.upper),
              [new_idx(idx1.lower, idx2.lower - 1)],
              [new_idx(idx1.upper + 1, idx2.upper)]
             )
    else :
      return (new_idx(idx2.lower, idx2.upper),
              [new_idx(idx1.lower, idx2.lower - 1)] + ([new_idx(idx2.upper + 1, idx1.upper)] if idx1.upper != idx2.upper else []),
              []
             )
  else :
    if idx2.upper < idx1.upper :
      return (new_idx(idx1.lower, idx2.upper),
              [new_idx(idx2.upper + 1, idx1.upper)],
              [new_idx(idx2.lower, idx1.lower - 1)] if  idx1.lower != idx2.lower else []
             )
    else :
      return (new_idx(idx1.lower, idx1.upper),
              [],
              ([new_idx(idx2.lower, idx1.lower - 1)] if idx1.lower != idx2.lower else []) + ([new_idx(idx1.upper + 1, idx2.upper)] if idx1.upper != idx2.upper else [])
             )

class _Full :
  def __init__(self, positions, indices) :
    self.dim = len(positions)
    assert self.dim == 1 or self.dim == 2
    assert len(indices) == self.dim
    if self.dim == 2 :
      assert get_master_idx(indices)[0] != get_master_idx(indices)[1]
      assert get_types(indices)[0] == get_types(indices)[1]
    self.positions = positions
    self.indices = indices
  def __str__(self) :
    return f'{srepr(self.indices)}'
  def __and__(self, other) :
    assert self.positions == other.positions
    if isinstance(other, _Full) :
      if self.dim == 1 :
        common, first, second = _intersect_idx(self.indices[0], other.indices[0])
        if common != None :
          return _Full(self.positions, (common,)), [_Full(self.positions, (f,)) for f in first], [_Full(self.positions, (s,)) for s in second]
        else :
          return None, [self], [other]
      else :
        common1, first1, second1 = _intersect_idx(self.indices[0], other.indices[0])
        common2, first2, second2 = _intersect_idx(self.indices[1], other.indices[1])
        if common1 != None and common2 != None :
          first = [_Full(self.positions, (f1, self.indices[1])) for f1 in first1]
          first.extend(_Full(self.positions, (common1, f2)) for f2 in first2)
          second = [_Full(self.positions, (s1, other.indices[1])) for s1 in second1]
          second.extend(_Full(self.positions, (common1, s2)) for s2 in second2)
          return _Full(self.positions, (common1, common2)), first, second
        else :
          # Тут бы копии делать, что имена индексов были другими... Наверно...
          return None, [self], [other]
    elif isinstance(other, _Rel) :
      common1, *_ = _intersect_idx(self.indices[0], other.indices[0])
      common2, *_ = _intersect_idx(self.indices[1], other.indices[1])
      if common1 != None and common2 != None :
        rel = _Rel.normalize(self.positions, (common1, common2), other.shift)
        if rel == None :
          return None, [self], [other]
        common1, first1, _ = _intersect_idx(self.indices[0], rel.indices[0])
        common2, first2, _ = _intersect_idx(self.indices[1], rel.indices[1])
        assert common1.lower == rel.indices[0].lower and common1.upper == rel.indices[0].upper
        assert common2.lower == rel.indices[1].lower and common2.upper == rel.indices[1].upper
        first = [_Full(self.positions, (f1, self.indices[1])) for f1 in first1]
        first.extend(_Full(self.positions, (common1, f2)) for f2 in first2)
        excl = ~rel
        if excl != None :
          first.append(excl)
        common1, _, second1 = _intersect_idx(rel.indices[0], other.indices[0])
        assert common1.lower == rel.indices[0].lower and common1.upper == rel.indices[0].upper
        second = [_Rel(self.positions, (s1, ShiftedIdx(s1, other.shift)), other.shift) for s1 in second1]
        return rel, first, second
      else :
        return None, [self], [other]
    elif isinstance(other, _Excl) :
      common1, first1, second1 = _intersect_idx(self.indices[0], other.indices[0])
      common2, first2, second2 = _intersect_idx(self.indices[1], other.indices[1])
      if common1 != None and common2 != None :
        excl = _Excl.normalize(self.positions, (common1, common2), other.excludes)
        if excl == None :
          return None, [self], [other]
        first = [_Full(self.positions, (f1, self.indices[1])) for f1 in first1]
        first.extend(_Full(self.positions, (common1, f2)) for f2 in first2)
        if isinstance(excl, _Excl) :
          first.extend(~excl)
        second_args = [(self.positions, (s1, other.indices[1]), other.excludes) for s1 in second1]
        second_args.extend((self.positions, (common1, s2), other.excludes) for s2 in second2)
        second = list(filter(None, (_Excl.normalize(*s_args) for s_args in second_args)))
        return excl, first, second
      else :
        return None, [self], [other]
    else :
      return NotImplemented

class _Rel :
  def __init__(self, positions, indices, shift) :
    assert len(positions) == 2
    assert len(indices) == 2
    assert get_master_idx(indices)[0] == get_master_idx(indices)[1]
    assert get_shifts(indices)[0] == 0
    assert shift == get_shifts(indices)[1]
    self.positions = positions
    self.indices = indices
    self.shift = shift
  def __str__(self) :
    return f'{srepr(self.indices)}:{self.shift}'
  @classmethod
  def normalize(cls, positions, indices, shift) :
    low = max(indices[0].lower, indices[1].lower - shift)
    up = min(indices[0].upper, indices[1].upper - shift)
    if up < low :
      return None
    idx = get_types(indices)[0](IDummy(), (low, up))
    return cls(positions, (idx, ShiftedIdx(idx, shift)), shift)
  def __invert__(self) :
    return _Excl.normalize(self.positions, (self.indices[0], block_copy((self.indices[1],))[0]), frozenset({self.shift}))
  def __and__(self, other) :
    assert self.positions == other.positions
    if isinstance(other, _Full) :
      common, first, second = other & self
      return common, second, first
    elif isinstance(other, _Rel) :
      if self.shift != other.shift :
        return None, [self], [other]
      else :
        common, first, second = _intersect_idx(self.indices[0], other.indices[0])
        if common != None :
          return _Rel(self.positions, (common, ShiftedIdx(common, self.shift)), self.shift), [_Rel(self.positions, (f, ShiftedIdx(f, self.shift)), self.shift) for f in first], [_Rel(self.positions, (s, ShiftedIdx(s, self.shift)), self.shift) for s in second]
        else :
          return None, [self], [other]
    elif isinstance(other, _Excl) :
      common1, *_ = _intersect_idx(self.indices[0], other.indices[0])
      common2, *_ = _intersect_idx(self.indices[1], other.indices[1])
      if common1 != None and common2 != None :
        if self.shift in other.excludes :
          return None, [self], [other]
        rel = _Rel.normalize(self.positions, (common1, common2), self.shift)
        if rel == None :
          return None, [self], [other]
        common1, first1, _ = _intersect_idx(self.indices[0], rel.indices[0])
        assert common1.lower == rel.indices[0].lower and common1.upper == rel.indices[0].upper
        common1, _, second1 = _intersect_idx(rel.indices[0], other.indices[0])
        common2, _, second2 = _intersect_idx(rel.indices[1], other.indices[1])
        assert common1.lower == rel.indices[0].lower and common1.upper == rel.indices[0].upper
        assert common2.lower == rel.indices[1].lower and common2.upper == rel.indices[1].upper
        first = [_Rel(self.positions, (f1, ShiftedIdx(f1, self.shift)), self.shift) for f1 in first1]
        second_args = [(self.positions, (common1, common2), other.excludes | frozenset({self.shift}))]
        second_args.extend((self.positions, (s1, other.indices[1]), other.excludes) for s1 in second1)
        second_args.extend((self.positions, (common1, s2), other.excludes) for s2 in second2)
        second = list(filter(None, (_Excl.normalize(*s_args) for s_args in second_args)))
        return rel, first, second
      else :
        return None, [self], [other]
    else :
      return NotImplemented

class _Excl :
  def __init__(self, positions, indices, excludes) :
    assert len(positions) == 2
    assert len(indices) == 2
    assert get_master_idx(indices)[0] != get_master_idx(indices)[1]
    assert get_types(indices)[0] == get_types(indices)[1]
    self.positions = positions
    self.indices = indices
    self.excludes = frozenset(excludes)
  def __str__(self) :
    return f'{srepr(self.indices)}:{self.excludes}'
  @classmethod
  def normalize(cls, positions, indices, excludes) :
    assert get_master_idx(indices)[0] != get_master_idx(indices)[1]
    assert get_types(indices)[0] == get_types(indices)[1]
    excludes = {shift for shift in excludes if max(indices[0].lower, indices[1].lower - shift) <= min(indices[0].upper, indices[1].upper - shift)}
    pos = 1 if len(indices[0]) == 1 else (0 if len(indices[1]) == 1 else None)
    if pos != None :
      indices = list(indices)
      while True :
        shift = indices[1].lower - indices[0].lower
        if shift in excludes and len(indices[pos]) != 1 :
          indices[pos] = get_types(indices)[pos](IDummy(), (indices[pos].lower + 1, indices[pos].upper))
          excludes.remove(shift)
        else :
          break
      while True :
        shift = indices[1].upper - indices[0].upper
        if shift in excludes and len(indices[pos]) != 1 :
          indices[pos] = get_types(indices)[pos](IDummy(), (indices[pos].lower, indices[pos].upper - 1))
          excludes.remove(shift)
        else :
          break
      indices = tuple(indices)
    if len(excludes) == len(indices[0]) + len(indices[1]) - 1 :
      return None
    elif excludes :
      return cls(positions, indices, excludes)
    else :
      return _Full(positions, indices)
  def __invert__(self) :
    return [_Rel.normalize(self.positions, self.indices, shift) for shift in sorted(self.excludes)]
  def __and__(self, other) :
    assert self.positions == other.positions
    if isinstance(other, _Full) :
      common, first, second = other & self
      return common, second, first
    elif isinstance(other, _Rel) :
      common, first, second = other & self
      return common, second, first
    elif isinstance(other, _Excl) :
      common1, first1, second1 = _intersect_idx(self.indices[0], other.indices[0])
      common2, first2, second2 = _intersect_idx(self.indices[1], other.indices[1])
      if common1 != None and common2 != None :
        excl = _Excl.normalize(self.positions, (common1, common2), self.excludes | other.excludes)
        if excl == None :
          return None, [self], [other]
        excl1 = _Excl.normalize(self.positions, (common1, common2), other.excludes - self.excludes)
        first = ~excl1 if isinstance(excl1, _Excl) else []
        first_args = [(self.positions, (f1, self.indices[1]), self.excludes) for f1 in first1]
        first_args.extend((self.positions, (common1, f2), self.excludes) for f2 in first2)
        first.extend(filter(None, (_Excl.normalize(*f_args) for f_args in first_args)))
        excl2 = _Excl.normalize(self.positions, (common1, common2), self.excludes - other.excludes)
        second = ~excl2 if isinstance(excl2, _Excl) else []
        second_args = [(self.positions, (s1, other.indices[1]), other.excludes) for s1 in second1]
        second_args.extend((self.positions, (common1, s2), other.excludes) for s2 in second2)
        second.extend(filter(None, (_Excl.normalize(*s_args) for s_args in second_args)))
        return excl, first, second
      else :
        return None, [self], [other]
    else :
      return NotImplemented

class Part :
  def __init__(self, *args) :
    if len(args) == 1 :
      other = args[0]
      assert type(other) == type(self)
      self.__block_id = other.__block_id
      self.__indices = other.__indices
      self.__term = other.__term
      self.__full = set(other.__full)
      self.__relations = dict(other.__relations)
      self.__excludes = dict(other.__excludes)
      return
    assert len(args) == 3
    block_id, indices, term = args
    self.__block_id = block_id
    self.__indices = tuple(indices)
    self.__term = term
    #self.__size = 1
    self.__full = set()
    self.__relations = {}
    self.__excludes = {}
    if not indices :
      return
    # Выясняем, на каких позициях стоят зависимые индексы
    unique_indices = defaultdict(list)
    # Проходим в порядке возрастания, поэтому получаем отсортированный список позиций
    master_indices = get_master_idx(indices)
    for n, idx in enumerate(master_indices) :
      unique_indices[idx].append(n)
    new_indices = list(indices)
    for positions in unique_indices.values() :
      if len(positions) == 1 :
        self.__full.add(positions[0])
      elif len(positions) == 2 :
        shifts = get_shifts(indices[pos] for pos in positions)
        idx = block_copy((indices[positions[0]],))[0]
        new_indices[positions[0]] = idx
        new_indices[positions[1]] = ShiftedIdx(idx, shifts[1] - shifts[0])
        self.__relations[tuple(positions)] = shifts[1] - shifts[0]
        self.__term = idx_subs(self.__term, indices[positions[0]], idx)
      else :
        assert False
      #self.__size *= len(indices[positions[0]])
    self.__indices = tuple(new_indices)
  def __str__(self) :
    return f'{srepr((self.__indices))}:{self.__term}:{self.__excludes}'
  @classmethod
  def _is_same(cls, p1, p2) :
    assert p1.__block_id == p2.__block_id
    if p1.__relations != p2.__relations or p1.__excludes != p2.__excludes :
      return False
    assert p1.__full == p2.__full
    term1 = p1.__term
    term2 = p2.__term
    indices = block_copy(p1.__indices)
    dependent = {rel[1] for rel in p1.__relations.keys()}
    # Попытка замены индексов здесь может привести к выходу из диапазона типа индекса, ловим IdxOutOfRangeError
    try :
      for n, (i, i1, i2) in enumerate(zip(indices, p1.__indices, p2.__indices)) :
        if n not in dependent :
          term1 = idx_subs(term1, i1, i)
          term2 = idx_subs(term2, i2, i)
    except IdxOutOfRangeError :
      return False
    if term1 != term2 :
      return False
    return True
  @classmethod
  def glue(cls, p1, p2) :
    ''' Пытаемся "склеить" две части.
        None -- нельзя "приклеить", иначе возвращаем новый "склеенный" блок. '''
    assert p1.block_id == p2.block_id
    if not cls._is_same(p1, p2) :
      return None
    idx1 = None
    # "Склеить" можно если ровно одна пара ведущих индексов имеет подряд идущие диапазоны, а остальные имееют одинаковые диапазоны
    for n, (i1, i2) in enumerate(zip(p1.__indices, p2.__indices)) :
      if i1.lower != i2.lower or i1.upper != i2.upper :
        if i1.lower == i2.upper + 1 or i2.lower == i1.upper + 1 :
          if idx1 == None :
            idx1 = get_types((i1,))[0](IDummy(), (min(i1.lower, i2.lower), max(i1.upper, i2.upper)))
            pos1 = n
            pos2 = None
            continue
          else :
            if pos2 == None :
              shift = p1.__relations.get((pos1, n), None)
              if shift != None :
                pos2 = n
                idx2 = ShiftedIdx(idx1, shift)
                continue
        return None
    # Тут не должно быть полностью совпадающих блоков
    assert idx1 != None
    glued = cls(p1)
    glued.__term = idx_subs(glued.__term, glued.__indices[pos1], idx1)
    glued.__indices = list(glued.__indices)
    glued.__indices[pos1] = idx1
    if pos2 != None :
      glued.__indices[pos2] = idx2
    glued.__indices = tuple(glued.__indices)
    return glued
  @property
  def block_id(self) :
    return self.__block_id
  @property
  def indices(self) :
    return self.__indices
  @property
  def term(self) :
    return self.__term
  def __len__(self) :
    size = _master_block_size(self.__indices)
    for (pos1, pos2), excludes in self.__excludes.items() :
      idx1, idx2 = self.__indices[pos1], self.__indices[pos2]
      for shift in excludes :
        l = min(idx1.upper, idx2.upper - shift) - max(idx1.lower, idx2.lower - shift) + 1
        assert l > 0
        size -= l
    return size
  def __add__(self, other) :
    if not isinstance(other, type(self)) :
      return NotImplemented
    assert self.__block_id == other.__block_id
    assert self.__indices == other.__indices
    assert self.__full == other.__full
    assert self.__relations == other.__relations
    assert self.__excludes == other.__excludes
    new_part = type(self)(self)
    new_part.__term = self.__term + other.__term
    return new_part
  def get_pos(self, positions) :
    positions = tuple(positions)
    data = positions, tuple(self.__indices[pos] for pos in positions)
    if len(positions) == 1 :
      pos, = positions
      if pos in self.__full :
        return _Full(*data)
      else :
        for (pos1, pos2), shift in self.__relations.items() :
          if pos1 == pos or pos2 == pos :
            return _Rel((pos1, pos2), (self.__indices[pos1], self.__indices[pos2]), shift)
        for (pos1, pos2), excludes in self.__excludes.items() :
          if pos1 == pos or pos2 == pos :
            return _Excl((pos1, pos2), (self.__indices[pos1], self.__indices[pos2]), excludes)
        assert False
    elif len(positions) == 2 :
      if positions in self.__relations :
        return _Rel(*data, self.__relations[positions])
      elif positions in self.__excludes :
        return _Excl(*data, self.__excludes[positions])
      else :
        assert all(pos in self.__full for pos in positions)
        return _Full(*data)
    else :
      assert False
  def set_pos(self, pd) :
    positions = pd.positions
    new_part = type(self)(self)
    if len(positions) == 2 :
      new_part.__full.difference_update(positions)
      new_part.__excludes.pop(positions, None)
      if isinstance(pd, _Full) :
        assert pd.positions not in new_part.__relations
        new_part.__full.update(positions)
      elif isinstance(pd, _Rel) :
        if pd.positions in new_part.__relations :
          assert new_part.__relations[positions] == pd.shift
        new_part.__relations[positions] = pd.shift
      elif isinstance(pd, _Excl) :
        assert pd.positions not in new_part.__relations
        new_part.__excludes[positions] = pd.excludes
      else :
        assert False
      indices = [new_part.__indices[pos] for pos in positions]
      master_indices = get_master_idx(indices)
      if master_indices[0] != master_indices[1] :
        tmp = block_copy(indices)[0]
        new_part.__term = idx_subs(new_part.__term, indices[0], tmp)
        new_part.__term = idx_subs(new_part.__term, indices[1], pd.indices[1])
        new_part.__term = idx_subs(new_part.__term, tmp, pd.indices[0])
      elif isinstance(pd, _Rel) :
        new_part.__term = idx_subs(new_part.__term, indices[0], pd.indices[0])
      else :
        assert False
      new_part.__indices = list(new_part.__indices)
      new_part.__indices[positions[0]] = pd.indices[0]
      new_part.__indices[positions[1]] = pd.indices[1]
      new_part.__indices = tuple(new_part.__indices)
    else :
      pos, = positions
      assert pos in new_part.__full
      new_part.__term = idx_subs(new_part.__term, new_part.__indices[pos], pd.indices[0])
      new_part.__indices = new_part.__indices[:pos] + (pd.indices[0],) + new_part.__indices[pos + 1:]
    return new_part
  def generate_loop(self, body, *, continue_cond = False) :
    excl_cond = False
    for (pos1, pos2), excludes in self.__excludes.items() :
      for shift in sorted(excludes) :
        excl_cond |= Eq(self.__indices[pos1], self.__indices[pos2] - shift)
    if excl_cond != False :
      body.insert(0, If(excl_cond, [continue_]))
    if continue_cond != False :
      body.insert(0, If(continue_cond, [continue_]))
    dependent = {pos2 for pos1, pos2 in self.__relations.keys()}
    loop_indices = [idx for n, idx in enumerate(self.__indices) if n not in dependent]
    body = wrap_in_loop(body, loop_indices)
    return body

def _intersect_part(part1, part2) :
  ''' Находим пересечение двух частей. Результат -- часть, являющаяся пересечением исходных,
      и, возможно пустые, списки частей, принадлежащих только одной из исходных частей. '''
  assert part1.block_id == part2.block_id
  first_parts = []
  second_parts = []
  common_part1 = part1
  common_part2 = part2
  # Пересекаем последовательно пары индексов, части, не попавшие в пересечение дополняем оставшимися полными индексами, и получаем части,
  # принадлежащие только одной из исходных
  # Набор всех пересечений даст общую часть
  all_positions = list(range(len(part1.indices) - 1, -1, -1))
  while all_positions :
    pos = all_positions.pop()
    pd1 = part1.get_pos((pos,))
    pd2 = part2.get_pos(pd1.positions)
    if len(pd1.positions) != len(pd2.positions) :
      pd1 = part1.get_pos(pd2.positions)
    if len(pd1.positions) == 2 :
      all_positions.remove(pd1.positions[1])
    common, first, second = pd1 & pd2
    if common != None :
      first_parts.extend(common_part1.set_pos(f) for f in first)
      second_parts.extend(common_part2.set_pos(s) for s in second)
      common_part1 = common_part1.set_pos(common)
      common_part2 = common_part2.set_pos(common)
    else :
      return None, [part1], [part2]
  return common_part1 + common_part2, first_parts, second_parts

def _glue(parts, new) :
  ''' Добавляем часть в набор частей, по возможности "cклеивая" части. '''
  def try_to_glue(new) :
    for n, p in enumerate(parts) :
      glued = Part.glue(p, new)
      if glued != None :
        parts.pop(n)
        if not try_to_glue(glued) :
          parts.append(glued)
        return True
    return False
  if not try_to_glue(new) :
    parts.append(new)

def to_disjoint_parts(parts) :
  # Строим набор непересекающихся частей, добавляя в него по одному части из исходного набора
  # и при необходимости перестраивая сам набор непересекающихся частей
  def construct_disjoint(parts, disjoint) :
    while parts :
      p = parts.pop()
      for n, d_p in enumerate(disjoint) :
        common, first, second = _intersect_part(p, d_p)
        if common != None :
          disjoint, disjoint_tail = disjoint[:n], disjoint[n + 1:]
          _glue(disjoint, common)
          for elem in construct_disjoint(first, disjoint_tail) :
            _glue(disjoint, elem)
          for s in second :
            _glue(disjoint, s)
          break
      else :
        _glue(disjoint, p)
    return disjoint
  return construct_disjoint(parts, [])

def cmp_with_diag(row_indices, col_indices) :
  ''' Возвращает набор из трех чисел: количество элементов под, на и над диаганалью. '''
  assert get_types(row_indices) == get_types(col_indices)
  if not row_indices :
    return (0, 1, 0)
  idx_r, idx_c = row_indices[0], col_indices[0]
  master = get_master_idx((idx_r, idx_c))
  if master[0] == master[1] :
    shifts = get_shifts((idx_r, idx_c))
    if shifts[0] > shifts[1] :
      return _master_block_size(row_indices + col_indices), 0, 0
    elif shifts[0] < shifts[1] :
      return 0, 0, _master_block_size(row_indices + col_indices)
    else :
      under, diag, over = cmp_with_diag(row_indices[1:], col_indices[1:])
      l = len(idx_r)
      return l * under, l * diag, l * over
  else :
    if idx_r.lower > idx_c.upper :
      return _master_block_size(row_indices + col_indices), 0, 0
    elif idx_r.upper < idx_c.lower :
      return 0, 0, _master_block_size(row_indices + col_indices)
    else :
      common, first, second = _intersect_idx(idx_r, idx_c)
      assert common != None
      diag = len(common)
      under = (diag**2 - diag) // 2
      over = under
      for f in first :
        if f.lower > common.upper :
          under += len(f) * len(idx_c)
        elif f.upper < common.lower :
          over += len(f) * len(idx_c)
        else :
          assert False
      for s in second :
        if s.lower > common.upper :
          over += len(common) * len(s)
        elif s.upper < common.lower :
          under += len(common) * len(s)
        else :
          assert False
      ret = cmp_with_diag(row_indices[1:], col_indices[1:])
      all_elems = sum(ret)
      under = all_elems * under + ret[0] * diag
      over = all_elems * over + ret[2] * diag
      diag *= ret[1]
      return under, diag, over

if __name__ == "__main__" :
  from sympy import Symbol
  from itertools import starmap
  from sympy2ipopt.idx_type import IdxType
  from sympy2ipopt.utils.test_utils import renum_dummy, check_limits
  from sympy.codegen.ast import Assignment

  t1 = IdxType('t1', (0, 10))
  t2 = IdxType('t2', (2, 8))
  t3 = IdxType('t3', (-5, 3))
  t4 = IdxType('t4', (-9, -4))

  m0 = t1('m0', (0, 9))
  sm0 = ShiftedIdx(m0, 1)
  m1 = t1('m1', (0, 7))
  m2 = t1('m2', (4, 10))
  m3 = t1('m3', (0, 3))
  m4 = t1('m4', (8, 10))
  m5 = t1('m5', (0, 5))
  m6 = t1('m6', (3, 7))
  m7 = t1('m7', (2, 5))
  m8 = t1('m8', (0, 7))
  m9 = t1('m9', (1, 8))
  assert renum_dummy(_intersect_idx(m1, m2)) == (t1('_Dummy_1', (4, 7)), [t1('_Dummy_2', (0, 3))], [t1('_Dummy_3', (8, 10))])
  assert _intersect_idx(m2, m3) == (None, [m2], [m3])
  assert _intersect_idx(m1, m4) == (None, [m1], [m4])
  assert renum_dummy(_intersect_idx(m1, m5)) == (t1('_Dummy_1', (0, 5)), [t1('_Dummy_2', (6, 7))], [])
  assert renum_dummy(_intersect_idx(m1, m6)) == (t1('_Dummy_1', (3, 7)), [t1('_Dummy_2', (0, 2))], [])
  assert renum_dummy(_intersect_idx(m1, m7)) == (t1('_Dummy_1', (2, 5)), [t1('_Dummy_2', (0, 1)), t1('_Dummy_3', (6, 7))], [])
  assert renum_dummy(_intersect_idx(m2, m1)) == (t1('_Dummy_1', (4, 7)), [t1('_Dummy_2', (8, 10))], [t1('_Dummy_3', (0, 3))])
  assert renum_dummy(_intersect_idx(m7, m1)) == (t1('_Dummy_1', (2, 5)), [], [t1('_Dummy_2', (0, 1)), t1('_Dummy_3', (6, 7))])
  assert renum_dummy(_intersect_idx(m5, m1)) == (t1('_Dummy_1', (0, 5)), [], [t1('_Dummy_2', (6, 7))])
  assert renum_dummy(_intersect_idx(m6, m1)) == (t1('_Dummy_1', (3, 7)), [], [t1('_Dummy_2', (0, 2))])
  assert renum_dummy(_intersect_idx(m1, m8)) == (t1('_Dummy_1', (0, 7)), [], [])
  assert renum_dummy(_intersect_idx(m1, m1)) == (t1('_Dummy_1', (0, 7)), [], [])
  assert renum_dummy(_intersect_idx(m8, m9)) == (t1('_Dummy_1', (1, 7)), [t1('_Dummy_2', (0, 0))], [t1('_Dummy_3', (8, 8))])
  assert renum_dummy(_intersect_idx(sm0, m1)) == (t1('_Dummy_1', (1, 7)), [t1('_Dummy_2', (8, 10))], [t1('_Dummy_3', (0, 0))])
  assert renum_dummy(_intersect_idx(sm0, t1('i1'))) == (t1('_Dummy_1', (1, 10)), [], [t1('_Dummy_2', (0, 0))])


  n1 = t3('n1', (-5, -1))
  n2 = t3('n2', (-3, 1))
  n3 = t3('n3', (-5, 1))
  n4 = t3('n4', (-3, -1))
  n5 = t3('n5', (0, 1))
  n6 = t3('n6', (-5, -4))

  f1 = _Full((0,), (n1,))
  assert f1.dim == 1 and f1.positions == (0,) and f1.indices == (n1,)
  f2 = _Full((0,), (n2,))
  common, first, second = f1 & f2
  assert common.dim == 1 and check_limits(common.indices, [(-3, -1)])
  assert len(first) == 1, first[0].dim == 1 and check_limits(first[0].indices, [(-5, -4)]) 
  assert len(second) == 1, second[0].dim == 1 and check_limits(second[0].indices, [(0, 1)]) 
  f3 = _Full((0,), (n5,))
  common, first, second = f3 & f1
  assert common == None and first == [f3] and second == [f1]

  t5 = IdxType('t5', (-10, 10))
  a1 = t5('a1', (0, 7))
  a2 = t5('a2', (4, 10))
  b1 = t5('b1', (-5, -1))
  b2 = t5('b2', (-3, 1))
  b3 = t5('b3', (-5, 1))
  b4 = t5('b4', (-3, -1))
  b5 = t5('b5', (0, 1))
  f1 = _Full((0, 1), (a1, b1))
  assert f1.dim == 2 and f1.positions == (0, 1) and f1.indices == (a1, b1)
  f2 = _Full((0, 1), (a2, b2))
  common, first, second = f1 & f2
  assert check_limits(common.indices, [(4, 7), (-3, -1)])
  assert len(first) == 2 and type(first[0]) == _Full and check_limits(first[0].indices, [(0, 3), (-5, -1)]) and type(first[1]) == _Full and check_limits(first[1].indices, [(4, 7), (-5, -4)])
  assert len(second) == 2 and type(second[0]) == _Full and check_limits(second[0].indices, [(8, 10), (-3, 1)]) and type(second[1]) == _Full and check_limits(second[1].indices, [(4, 7), (0, 1)])

  f1 = _Full((0, 1), (a1, b3))
  f2 = _Full((0, 1), (a2, b4))
  common, first, second = f1 & f2
  assert check_limits(common.indices, [(4, 7), (-3, -1)])
  assert len(first) == 3 and check_limits(first[0].indices, [(0, 3), (-5, 1)]) and check_limits(first[1].indices, [(4, 7), (-5, -4)]) and check_limits(first[2].indices, [(4, 7), (0, 1)])
  assert len(second) == 1 and check_limits(second[0].indices, [(8, 10), (-3, -1)])

  f1 = _Full((0, 1), (a1, b1))
  f2 = _Full((0, 1), (a2, b5))
  common, first, second = f1 & f2
  assert common == None and first == [f1] and second == [f2]
  
  r1 = _Rel((0, 1), (m0, sm0), 1)
  assert r1.positions == (0, 1) and r1.indices == (m0, sm0) and r1.shift == 1
  r2 = _Rel.normalize((0, 1), (m1, m2), -2)
  assert r2.positions == (0, 1) and check_limits(r2.indices, [(6, 7), (4, 5)]) and r2.shift == -2
  e1 = ~r2
  assert e1.positions == r2.positions and frozenset({r2.shift}) == e1.excludes and check_limits(e1.indices, [(6, 7), (4, 5)])
  r3 = _Rel.normalize((0, 1), (m1, m2), -3)
  assert ~r3 == None

  r = _Rel.normalize((0, 1), (m1, m2), -2)
  f = _Full((0, 1), (m1, m3))
  common, first, second = r & f
  assert common == None and first == [r] and second == [f]

  r = _Rel.normalize((0, 1), (m1, m2), -2)
  m10 = t1('m10', (5, 6))
  m11 = t1('m11', (5, 6))
  f = _Full((0, 1), (m10, m11))
  common, first, second = r & f
  assert common == None and first == [r] and second == [f]

  r = _Rel.normalize((0, 1), (m1, m2), -2)
  f = _Full((0, 1), (m10, m2))
  common, first, second = f & r
  assert check_limits(common.indices, [(6, 6), (4, 4)]) and common.shift == -2
  assert len(first) == 2 and type(first[0]) == _Full and check_limits(first[0].indices, [(5, 5), (4, 10)]) and type(first[1]) == _Full and check_limits(first[1].indices, [(6, 6), (5, 10)])
  assert len(second) == 1 and check_limits(second[0].indices, [(7, 7), (5, 5)]) and second[0].shift == -2

  r = _Rel.normalize((0, 1), (m1, m2), -2)
  f = _Full((0, 1), (m1, m2))
  common, first, second = f & r
  assert check_limits(common.indices, [(6, 7), (4, 5)]) and common.shift == -2
  assert len(first) == 3 and type(first[0]) == _Full and check_limits(first[0].indices, [(0, 5), (4, 10)]) and type(first[1]) == _Full and check_limits(first[1].indices, [(6, 7), (6, 10)]) and type(first[2]) == _Excl and check_limits(first[2].indices, [(6, 7), (4, 5)]) and first[2].excludes == frozenset({-2})
  assert second == []

  f = _Full((0, 1), (m5, m7))
  r = _Rel.normalize((0, 1), (m0, m1), 1)
  common, first, second = f & r
  assert common.shift == 1 and check_limits(common.indices, [(1, 4), (2, 5)])
  assert len(first) == 3 and type(first[0]) == _Full and check_limits(first[0].indices, [(0 ,0), (2, 5)]) and check_limits(first[1].indices, [(5, 5), (2, 5)]) and first[2].excludes == frozenset({1}) and check_limits(first[2].indices, [(1, 4), (2, 5)])
  assert len(second) == 2 and second[0].shift == 1 and check_limits(second[0].indices, [(0, 0), (1, 1)]) and second[1].shift == 1 and check_limits(second[1].indices, [(5, 6), (6, 7)])

  e2 = _Excl((0, 1), (m1, m2), {-2, 1})
  assert e2.positions == (0, 1) and e2.indices == (m1, m2) and e2.excludes == frozenset({-2, 1})
  e3 = _Excl.normalize((0, 1), (m1, m2), {-2, 1})
  assert e2.positions == (0, 1) and e2.indices == (m1, m2) and e2.excludes == frozenset({-2, 1})
  e4 = _Excl.normalize((0, 1), (m1, m2), {-4, -2, 1, 11})
  assert e2.positions == (0, 1) and e2.indices == (m1, m2) and e2.excludes == frozenset({-2, 1})
  e5 = _Excl.normalize((0, 1), (m1, m2), {-4, 11})
  assert type(e5) == _Full and e5.indices == (m1, m2)
  e6 = _Excl.normalize((0, 1), (m10, m11), {-1, 0, 1})
  assert e6 == None
  e7 = _Excl.normalize((0, 1), (m10, m11), {-1, 1})
  rels = ~e7
  assert len(rels) == 2 and type(rels[0]) == _Rel and rels[0].shift == -1 and check_limits(rels[0].indices, [(6, 6), (5, 5)]) and type(rels[1]) == _Rel and rels[1].shift == 1 and check_limits(rels[1].indices, [(5, 5), (6, 6)]) and rels[0].positions == e7.positions and rels[1].positions == e7.positions
  p1 = t1('p1', (0, 0))
  e8 = _Excl.normalize((0, 1), (p1, m0), {0, 1, 2})
  assert type(e8) == _Full and e8.positions == (0, 1) and renum_dummy(e8.indices) == (p1, t1('_Dummy_1', (3, 9)))
  e9 = _Excl.normalize((0, 1), (m0, p1), {0, -1, -2})
  assert type(e9) == _Full and e9.positions == (0, 1) and renum_dummy(e9.indices) == (t1('_Dummy_1', (3, 9)), p1)
  e10 = _Excl.normalize((0, 1), (m0, p1), {0, -1, -2, -9, -8})
  assert type(e10) == _Full and e10.positions == (0, 1) and renum_dummy(e10.indices) == (t1('_Dummy_1', (3, 7)), p1)
  e11 = _Excl.normalize((0, 1), (p1, m0), {0, 1, 2, 9, 8})
  assert type(e11) == _Full and e11.positions == (0, 1) and renum_dummy(e11.indices) == (p1, t1('_Dummy_1', (3, 7)))
  e12 = _Excl.normalize((0, 1), (m0, p1), {0, -1, -2, -5, -9, -8})
  assert e12.excludes == frozenset({-5}) and e12.positions == (0, 1) and renum_dummy(e12.indices) == (t1('_Dummy_1', (3, 7)), p1)

  e = _Excl((0, 1), (m1, m2), {-2, 1})
  f = _Full((0, 1), (m1, m3))
  common, first, second = e & f
  assert common == None and first == [e] and second == [f]

  e = _Excl((0, 1), (m1, m2), {-2, 1})
  f = _Full((0, 1), (m1, m3))
  common, first, second = e & f
  assert common == None and first == [e] and second == [f]

  f = _Full((0, 1), (m1, m2))
  m12 = t1('m12', (7, 8))
  m13 = t1('m13', (3, 4))
  e = _Excl((0, 1), (m12, m13), {-4})
  common, first, second = e & f
  assert type(common) == _Full and check_limits(common.indices, [(7, 7), (4, 4)])
  assert len(first) == 1 and type(first[0]) == _Full and check_limits(first[0].indices, [(8, 8), (3, 3)])
  assert len(second) == 2 and type(second[0]) == _Full and check_limits(second[0].indices, [(0, 6), (4, 10)]) and type(second[1]) == _Full and check_limits(second[1].indices, [(7, 7), (5, 10)])

  f = _Full((0, 1), (m1, m9))
  m14 = t1('m14', (8, 9))
  e = _Excl((0, 1), (m12, m14), {1})
  common, first, second = f & e
  assert common == None and first == [f] and second == [e]

  f = _Full((0, 1), (m1, m2))
  e = _Excl((0, 1), (m10, m11), {0})
  common, first, second = e & f
  assert type(common) == _Excl and check_limits(common.indices, [(5, 6), (5, 6)]) and common.excludes == frozenset({0})
  assert first == []
  assert len(second) == 5 and check_limits(second[0].indices, [(0, 4), (4, 10)]) and check_limits(second[1].indices, [(7, 7), (4, 10)]) and check_limits(second[2].indices, [(5, 6), (4, 4)]) and check_limits(second[3].indices, [(5, 6), (7, 10)]) and type(second[4]) == _Rel and check_limits(second[4].indices, [(5, 6), (5, 6)]) and second[4].shift == 0

  r1 = _Rel.normalize((0, 1), (m1, m2), 1)
  r2 = _Rel.normalize((0, 1), (m1, m2), -2)
  common, first, second = r1 & r2
  assert common == None and first == [r1] and second == [r2]

  r1 = _Rel.normalize((0, 1), (m1, m2), 0)
  r2 = _Rel.normalize((0, 1), (m3, m5), 0)
  common, first, second = r1 & r2
  assert common == None and first == [r1] and second == [r2]

  r1 = _Rel.normalize((0, 1), (m0, m1), 1)
  r2 = _Rel.normalize((0, 1), (m3, m7), 1)
  common, first, second = r1 & r2
  assert common.positions == (0, 1) and common.shift == 1 and check_limits(common.indices, [(1, 3), (2, 4)])
  assert len(first) == 2 and first[0].shift == 1 and check_limits(first[0].indices, [(0, 0), (1, 1)]) and first[1].shift == 1 and check_limits(first[1].indices, [(4, 6), (5, 7)])
  assert second == []

  e = _Excl((0, 1), (m1, m2), {-2, 1})
  r = _Rel.normalize((0, 1), (m1, m3), 0)
  common, first, second = e & r
  assert common == None and first == [e] and second == [r]

  e = _Excl((0, 1), (m1, m2), {-2, 1})
  r = _Rel.normalize((0, 1), (m1, m2), 1)
  common, first, second = r & e
  assert common == None and first == [r] and second == [e]

  e = _Excl((0, 1), (m1, m2), {-2, 1})
  r = _Rel.normalize((0, 1), (m12, m13), -4)
  common, first, second = e & r
  assert common == None and first == [e] and second == [r]

  e = _Excl((0, 1), (m5, m7), {0})
  r = _Rel.normalize((0, 1), (m0, m1), 1)
  common, first, second = e & r
  assert common.shift == 1 and check_limits(common.indices, [(1, 4), (2, 5)])
  assert len(first) == 3 and check_limits(first[0].indices, [(1, 4), (2, 5)]) and first[0].excludes == frozenset({0, 1}) and type(first[1]) == _Full and check_limits(first[1].indices, [(0 ,0), (2, 5)]) and type(first[2]) == _Full and check_limits(first[2].indices, [(5, 5), (2, 4)])
  assert len(second) == 2 and second[0].shift == 1 and check_limits(second[0].indices, [(0, 0), (1, 1)]) and second[1].shift == 1 and check_limits(second[1].indices, [(5, 6), (6, 7)])

  r = _Rel.normalize((0, 1), (m1, m2), -2)
  e = _Excl((0, 1), (m10, m2), {-1})
  common, first, second = e & r
  assert check_limits(common.indices, [(6, 6), (4, 4)]) and common.shift == -2
  assert len(first) == 2 and type(first[0]) == _Full and check_limits(first[0].indices, [(5, 5), (5, 10)]) and type(first[1]) == _Full and check_limits(first[1].indices, [(6, 6), (6, 10)])
  assert len(second) == 1 and check_limits(second[0].indices, [(7, 7), (5, 5)]) and second[0].shift == -2

  e1 = _Excl((0, 1), (m1, m2), {-2, 1})
  e2 = _Excl((0, 1), (m1, m3), {0})
  common, first, second = e1 & e2
  assert common == None and first == [e1] and second == [e2]

  e1 = _Excl((0, 1), (m10, m11), {0, 1})
  e2 = _Excl((0, 1), (m10, m11), {-1})
  common, first, second = e1 & e2
  assert common == None and first == [e1] and second == [e2]

  e1 = _Excl((0, 1), (m3, m5), {0, -1})
  e2 = _Excl((0, 1), (m9, m5), {1, 0})
  common, first, second = e1 & e2
  assert common.excludes == frozenset({-1, 1, 0}) and check_limits(common.indices, [(1, 3), (0, 5)])
  assert len(first) == 2 and first[0].shift == 1 and check_limits(first[0].indices, [(1, 3), (2, 4)]) and type(first[1]) == _Full and check_limits(first[1].indices, [(0, 0), (1, 5)])
  assert len(second) == 2 and second[0].shift == -1 and check_limits(second[0].indices, [(1, 3), (0, 2)]) and second[1].excludes == frozenset({0, 1}) and check_limits(second[1].indices, [(4, 8), (0, 5)])

  e1 = _Excl((0, 1), (m3, m5), {0, -1})
  e2 = _Excl((0, 1), (m9, m6), {1, 2})
  common, first, second = e1 & e2
  assert common.excludes == frozenset({0, 1, 2}) and check_limits(common.indices, [(1, 3), (3, 5)])
  assert len(first) == 4 and first[0].shift == 1 and check_limits(first[0].indices, [(2, 3), (3, 4)]) and first[1].shift == 2 and check_limits(first[1].indices, [(1,3), (3, 5)]) and type(first[2]) == _Full and check_limits(first[2].indices, [(0, 0), (1, 5)]) and first[3].excludes == frozenset({-1, 0}) and check_limits(first[3].indices, [(1, 3), (0, 2)])
  assert len(second) == 3 and second[0].shift == 0 and check_limits(second[0].indices, [(3, 3), (3, 3)]) and second[1].excludes == frozenset({1, 2}) and check_limits(second[1].indices, [(4, 8), (3, 7)]) and type(second[2]) == _Full and check_limits(second[2].indices, [(1, 3), (6, 7)])

  p = Part((Symbol('x'), Symbol('y')), (), Symbol('x') + Symbol('y'))
  assert p.block_id == (Symbol('x'), Symbol('y')) and p.indices == () and p.term == Symbol('x') + Symbol('y')

  p = Part(None, (m1, n1), m1 + n1)
  assert p.indices == (m1, n1) and p.term == m1 + n1
  p1 = Part(p)
  assert p1.indices == (m1, n1) and p1.term == m1 + n1

  p = Part(None, (m0, n1, sm0), m0**2)
  assert check_limits(p.indices, [(0, 9), (-5, -1), (1, 10)]) and get_master_idx(p.indices)[0] == get_master_idx(p.indices)[2] and get_shifts(p.indices) == (0, 0, 1) and p.indices[1] == n1 and p.term == p.indices[0]**2

  p = Part(None, (m0, n1, sm0, n1), m0**2 + n1)
  assert check_limits(p.indices, [(0, 9), (-5, -1), (1, 10), (-5, -1)]) and get_master_idx(p.indices)[0] == get_master_idx(p.indices)[2] and get_shifts(p.indices) == (0, 0, 1, 0) and get_master_idx(p.indices)[1] == get_master_idx(p.indices)[3] and renum_dummy(p.term) == t1('_Dummy_1', (0, 9))**2 + t3('_Dummy_2', (-5, -1))

  p1 = Part(None, (m0, n1, sm0), sm0**2)
  m15 = t1('m15', (0, 9))
  sm15 = ShiftedIdx(m15, 1)
  p2 = Part(None, (m15, sm15, n1), sm15**2)
  p3= Part(None, (m1, m2, n3), m1**2)
  p4 = Part(None, (m15, n1, sm15), sm15**2)
  p5 = Part(None, (m15, n1, sm15), sm15)
  assert Part._is_same(p1, p2) == False
  assert Part._is_same(p1, p3) == False
  assert Part._is_same(p1, p4) == True
  assert Part._is_same(p1, p5) == False
  p1 = Part(None, (m1, n1, m2), m1**2)
  p2 = Part(None, (m2, n1, m1), m2**2)
  assert Part._is_same(p1, p2) == True

  m0e = t1('m0e', (10, 10))
  p1 = Part(None, (m0,), sm0**2)
  p2 = Part(None, (m0e,), m0e**2)
  assert Part._is_same(p1, p2) == False
  assert Part._is_same(p2, p1) == False

  p1 = Part(None, (m1, n1, a1), m1)
  p2 = Part(None, (m1, n1, a1), n1)
  assert (p1 + p2).term == m1 + n1 and (p1 + p2).indices == (m1, n1, a1)

  p = Part(None, (m1, n1, a1), m1 + n1 + a1)
  p = p.set_pos(_Full((1,), (n2,)))
  assert p.indices == (m1, n2, a1) and p.term == m1 + n2 + a1

  p = Part(None, (m1, n1, m2), m1**2 + n1 + m2)
  p = p.set_pos(_Full((0, 2), (m2, m1)))
  assert p.indices == (m2, n1, m1) and p.term == m2**2 + n1 + m1

  p = Part(None, (m1, n1, m2), m1 + n1 + m2)
  p = p.set_pos(_Rel((0, 2), (m0, sm0), 1))
  assert p.indices == (m0, n1, sm0) and p.term == m0 + n1 + sm0
  p1 = Part(None, (m0, n1, sm0), m0 + n1 + sm0)
  assert Part._is_same(p, p1)

  p = Part(None, (m1, n1, m2), m1 + n1 + m2)
  p = p.set_pos(_Excl((0, 2), (m3, m5), {0}))
  assert p.indices == (m3, n1, m5) and p.term == m3 + n1 + m5
  pd = p.get_pos((0,))
  assert pd.positions == (0, 2) and pd.indices == (m3, m5) and pd.excludes == frozenset({0})

  p = Part(None, (m0, n1, sm0), m0**2 + n1 + sm0)
  m16 = t1('m16', (0, 7))
  sm16 = ShiftedIdx(m16, 1)
  p = p.set_pos(_Rel((0, 2), (m16, sm16), 1))
  assert p.indices == (m16, n1, sm16) and p.term == m16**2 + n1 + sm16
  
  p = Part(None, (m1, n1, m2), m1**2 + n1 + m2)
  p = p.set_pos(_Excl((0, 2), (m3, m5), {0}))
  pd = p.get_pos((0,))
  assert pd.positions == (0, 2) and pd.indices == (m3, m5) and pd.excludes == frozenset({0})
  p = p.set_pos(_Rel((0, 2), (m16, sm16), 1))
  assert p.indices == (m16, n1, sm16) and p.term == m16**2 + n1 + sm16

  p = Part(None, (m1, n1, m2), m1**2 + n1 + m2)
  p = p.set_pos(_Excl((0, 2), (m3, m5), {0}))
  pd = p.get_pos((0,))
  assert pd.positions == (0, 2) and pd.indices == (m3, m5) and pd.excludes == frozenset({0})
  p = p.set_pos(_Excl((0, 2), (m5, m8), {1, 0}))
  pd = p.get_pos((0,))
  assert pd.positions == (0, 2) and pd.indices == (m5, m8) and pd.excludes == frozenset({1, 0})
  assert p.indices == (m5, n1, m8) and p.term == m5**2 + n1 + m8

  p = Part(None, (m1, n1, m2), m1**2 + n1 + m2)
  pd = p.get_pos((0, 2))
  assert pd.positions == (0, 2) and pd.indices == (m1, m2) and type(pd) == _Full
  pd = p.get_pos((0,))
  assert pd.positions == (0,) and pd.indices == (m1,) and type(pd) == _Full
  
  p = Part(None, (m0, n1, sm0), m0**2 + n1 + sm0)
  pd = p.get_pos((0, 2))
  assert pd.positions == (0, 2) and check_limits(pd.indices, [(0, 9), (1, 10)]) and pd.shift == 1
  pd = p.get_pos((0,))
  assert pd.positions == (0, 2) and check_limits(pd.indices, [(0, 9), (1, 10)]) and pd.shift == 1
  pd = p.get_pos((2,))
  assert pd.positions == (0, 2) and check_limits(pd.indices, [(0, 9), (1, 10)]) and pd.shift == 1

  p = Part(None, (m1, n1, m2), m1**2 + n1 + m2)
  p = p.set_pos(_Excl((0, 2), (m3, m5), {0}))
  pd = p.get_pos((0, 2))
  assert pd.positions == (0, 2) and pd.indices == (m3, m5) and pd.excludes == frozenset({0})
  pd = p.get_pos((0,))
  assert pd.positions == (0, 2) and pd.indices == (m3, m5) and pd.excludes == frozenset({0})
  pd = p.get_pos((2,))
  assert pd.positions == (0, 2) and pd.indices == (m3, m5) and pd.excludes == frozenset({0})

  p = Part(None, (m1, n1, m2), m1**2 + n1 + m2)
  p = p.set_pos(_Excl((0, 2), (m3, m5), {0}))
  p1 = Part(p)
  pd = p1.get_pos((0,))
  assert pd.positions == (0, 2) and pd.indices == (m3, m5) and pd.excludes == frozenset({0})

  k1 = t2('k1')
  k2 = t2('k2', (4, 6))
  sk2 = ShiftedIdx(k2, -1)
  k3 = t2('k3', (7, 8))
  sk3 = ShiftedIdx(k3, -1)

  p1 = Part(None, (m1, n1, k1), m1 + n1)
  p2 = Part(None, (m2, n2, k2), m2 + n2)
  assert Part.glue(p1, p2) == None

  p1 = Part(None, (m1, n4, k2), m1 + k2)
  p2 = Part(None, (m4, n5, k3), m4 + k3)
  assert Part.glue(p1, p2) == None

  p1 = Part(None, (n1, m1), m1 + n1)
  p2 = Part(None, (n1, m4), m4 + n1)
  glued = Part.glue(p1, p2)
  assert renum_dummy(glued.indices) == (n1, t1('_Dummy_1', (0, 10))) and renum_dummy(glued.term) == t1('_Dummy_1', (0, 10)) + n1

  p1 = Part(None, (k1, n4, m1), k1 + n4)
  p2 = Part(None, (k1, n5, m1), k1 + n5)
  glued = Part.glue(p1, p2)
  assert renum_dummy(glued.indices) == (k1, t3('_Dummy_1', (-3, 1)), m1) and renum_dummy(glued.term) == k1 + t3('_Dummy_1', (-3, 1))
  glued = Part.glue(p2, p1)
  assert renum_dummy(glued.indices) == (k1, t3('_Dummy_1', (-3, 1)), m1) and renum_dummy(glued.term) == k1 + t3('_Dummy_1', (-3, 1))

  p1 = Part(None, (k1, n4, m1), k1)
  p2 = Part(None, (k1, n5, m4), k1)
  assert Part.glue(p1, p2) == None

  p1 = Part(None, (k2, sk2, n4), k2)
  p2 = Part(None, (k3, sk3, n5), k3)
  assert Part.glue(p1, p2) == None

  p1 = Part(None, (k2, m1, sk2), k2 + m1)
  p2 = Part(None, (k3, m1, sk3), k3 + m1)
  glued = Part.glue(p1, p2)
  assert renum_dummy(glued.indices) == (t2('_Dummy_1', (4, 8)), m1, ShiftedIdx(t2('_Dummy_1', (4, 8)), -1)) and renum_dummy(glued.term) == t2('_Dummy_1', (4, 8)) + m1

  p1 = Part(None, (k2, m1, sk2), k2 + m1)
  p2 = Part(None, (k3, m1, k3), k3 + m1)
  assert Part.glue(p1, p2) == None

  p1 = Part(None, (m1, n1), m1 * n1)
  p2 = Part(None, (m2, n2), m2 * n2)
  p3 = Part(None, (m3, n3), m3 * n3)
  p4 = Part(None, (m4, n4), m4 * n4)
  parts = [p1, p2, p3]
  _glue(parts, p4)
  assert parts == [p1, p2, p3, p4]

  parts = [p1, p2, p3]
  p5 = Part(None, (m4, n1), m4 * n1)
  _glue(parts, p5)
  assert len(parts) == 3 and parts[:2] == [p2, p3] and renum_dummy(parts[2].indices) == (t1('_Dummy_1', (0, 10)), n1) and renum_dummy(parts[2].term) == t1('_Dummy_1', (0, 10)) * n1

  p6 = Part(None, (m1, n1), m1 + n1)
  parts = [p6, p2, p3]
  _glue(parts, p5)
  assert parts == [p6, p2, p3, p5]

  parts = [p1, p4]
  p7 = Part(None, (m4, n6), m4 * n6)
  _glue(parts, p7)
  assert len(parts) == 1 and renum_dummy(parts[0].indices) == (t1('_Dummy_1', (0, 10)), n1) and renum_dummy(parts[0].term) == t1('_Dummy_1', (0, 10)) * n1

  p8 = Part(None, (m1, n1, k3), m1 * n1 + k3)
  p9 = Part(None, (m2, n2, k2), m2 * n2 + k2)
  p10 = Part(None, (m3, n3, k1), m3 * n3 + k1)
  parts = [p8, p9, p10]
  p11 = Part(None, (m4, n1, k2), m4 * n1 + k2)
  _glue(parts, p11)
  assert parts == [p8, p9, p10, p11]

  p12 = Part(None, (k2, n1, sk2), n1 + k2)
  parts = [p8, p12, p9]
  p13 = Part(None, (k3, n1, sk3), n1 + k3)
  _glue(parts, p13)
  k4 = t2('_Dummy_1', (4, 8))
  sk4 = ShiftedIdx(k4, -1)
  assert len(parts) == 3 and parts[:2] == [p8, p9] and renum_dummy(parts[2].indices) == (k4, n1, sk4)

  p1 = Part(None, (m1, n1), m1 + n1)
  p2 = Part(None, (m2, n2), m2 * n2)
  common, first, second = _intersect_part(p1, p2)
  assert renum_dummy(common.indices) == (t1('_Dummy_1', (4, 7)), t3('_Dummy_4', (-3, -1))) and renum_dummy(common.term) == t1('_Dummy_1', (4, 7)) + t3('_Dummy_4', (-3, -1)) + t1('_Dummy_1', (4, 7)) * t3('_Dummy_4', (-3, -1))
  assert len(first) == 2 and renum_dummy(first[0].indices) == (t1('_Dummy_1', (0, 3)), n1) and renum_dummy(first[1].indices) == (t1('_Dummy_1', (4, 7)), t3('_Dummy_5', (-5, -4))) and renum_dummy(first[0].term) == t1('_Dummy_1', (0, 3)) + n1 and renum_dummy(first[1].term) == t1('_Dummy_1', (4, 7)) + t3('_Dummy_5', (-5, -4))
  assert len(second) == 2 and renum_dummy(second[0].indices) == (t1('_Dummy_1', (8, 10)), n2) and renum_dummy(second[1].indices) == (t1('_Dummy_1', (4, 7)), t3('_Dummy_6', (0, 1))) and renum_dummy(second[0].term) == t1('_Dummy_1', (8, 10)) * n2 and renum_dummy(second[1].term) == t1('_Dummy_1', (4, 7)) * t3('_Dummy_6', (0, 1)) 


  p1 = Part(None, (m1, n3), m1 + n3)
  p2 = Part(None, (m2, n4), m2 * n4)
  common, first, second = _intersect_part(p1, p2)
  assert renum_dummy(common.indices) == (t1('_Dummy_1', (4, 7)), t3('_Dummy_4', (-3, -1))) and renum_dummy(common.term) == t1('_Dummy_1', (4, 7)) + t3('_Dummy_4', (-3, -1)) + t1('_Dummy_1', (4, 7)) * t3('_Dummy_4', (-3, -1))
  assert len(first) == 3 and renum_dummy(first[0].indices) == (t1('_Dummy_1', (0, 3)), n3) and renum_dummy(first[1].indices) == (t1('_Dummy_1', (4, 7)), t3('_Dummy_5', (-5, -4))) and renum_dummy(first[2].indices) == (t1('_Dummy_1', (4, 7)), t3('_Dummy_6', (0, 1))) and renum_dummy(first[0].term) == t1('_Dummy_1', (0, 3)) + n3 and renum_dummy(first[1].term) == t1('_Dummy_1', (4, 7)) + t3('_Dummy_5', (-5, -4)) and renum_dummy(first[2].term) == t1('_Dummy_1', (4, 7)) + t3('_Dummy_6', (0, 1))
  assert len(second) == 1 and renum_dummy(second[0].indices) == (t1('_Dummy_1', (8, 10)), n4) and renum_dummy(second[0].term) == t1('_Dummy_1', (8, 10)) * n4

  p1 = Part(None, (m1, n1, k1), m1 + n1 + k1)
  p2 = Part(None, (m3, n4, k2), m3 * n4 * k2)
  common, first, second = _intersect_part(p1, p2)
  assert renum_dummy(common.indices) == (t1('_Dummy_1', (0, 3)), t3('_Dummy_3', (-3, -1)), t2('_Dummy_5', (4, 6))) and renum_dummy(common.term) == t1('_Dummy_1', (0, 3)) + t3('_Dummy_3', (-3, -1)) + t2('_Dummy_5', (4, 6)) + t1('_Dummy_1', (0, 3)) * t3('_Dummy_3', (-3, -1)) * t2('_Dummy_5', (4, 6))
  assert len(first) == 4 and renum_dummy(first[0].indices) == (t1('_Dummy_1', (4, 7)), n1, k1) and renum_dummy(first[1].indices) == (t1('_Dummy_1', (0, 3)), t3('_Dummy_4', (-5, -4)), k1) and renum_dummy(first[2].indices) == (t1('_Dummy_1', (0, 3)), t3('_Dummy_3', (-3, -1)), t2('_Dummy_6', (2, 3))) and renum_dummy(first[3].indices) == (t1('_Dummy_1', (0, 3)), t3('_Dummy_3', (-3, -1)), t2('_Dummy_7', (7, 8))) and renum_dummy(first[0].term) == t1('_Dummy_1', (4, 7)) + n1 + k1 and renum_dummy(first[1].term) == t1('_Dummy_1', (0, 3)) + t3('_Dummy_4', (-5, -4)) + k1 and renum_dummy(first[2].term) == t1('_Dummy_1', (0, 3)) + t3('_Dummy_3', (-3, -1)) + t2('_Dummy_6', (2, 3)) and renum_dummy(first[3].term) == t1('_Dummy_1', (0, 3)) + t3('_Dummy_3', (-3, -1)) + t2('_Dummy_7', (7, 8))
  assert second == []

  p1 = Part(None, (m1, n1, k3), m1 + n1 + k3)
  p2 = Part(None, (m3, n4, k2), m3 * n4 * k2)
  assert _intersect_part(p1, p2) == (None, [p1], [p2])

  m17 = t1('m17', (3, 7))
  sm17 = ShiftedIdx(m17, -1)
  p1 = Part(None, (m0, n1, sm0), m0 + n1)
  p2 = Part(None, (sm17, n2, m17), m17 * n2)
  common, first, second = _intersect_part(p1, p2)
  assert renum_dummy(common.indices) == (t1('_Dummy_1', (2, 6)), t3('_Dummy_4', (-3, -1)), ShiftedIdx(t1('_Dummy_1', (2, 6)), 1)) and renum_dummy(common.term) == t1('_Dummy_1', (2, 6)) + t3('_Dummy_4', (-3, -1)) + ShiftedIdx(t1('_Dummy_1', (2, 6)), 1) * t3('_Dummy_4', (-3, -1))
  assert len(first) == 3 and renum_dummy(first[0].indices) == (t1('_Dummy_1', (0, 1)), n1, ShiftedIdx(t1('_Dummy_1', (0, 1)), 1)) and renum_dummy(first[1].indices) == (t1('_Dummy_1', (7, 9)), n1, ShiftedIdx(t1('_Dummy_1', (7, 9)), 1)) and renum_dummy(first[2].indices) == (t1('_Dummy_1', (2, 6)), t3('_Dummy_5', (-5, -4)), ShiftedIdx(t1('_Dummy_1', (2, 6)), 1)) and renum_dummy(first[0].term) == t1('_Dummy_1', (0, 1)) + n1 and renum_dummy(first[1].term) == t1('_Dummy_1', (7, 9)) + n1 and renum_dummy(first[2].term) == t1('_Dummy_1', (2, 6)) + t3('_Dummy_5', (-5, -4))
  assert len(second) == 1 and renum_dummy(second[0].indices) == (t1('_Dummy_1', (2, 6)), t3('_Dummy_6', (0, 1)), ShiftedIdx(t1('_Dummy_1', (2, 6)), 1)) and renum_dummy(second[0].term) == ShiftedIdx(t1('_Dummy_1', (2, 6)), 1) * t3('_Dummy_6', (0, 1))
  
  i1 = t1('i1')
  i3 = t3('i3')
  n7 = t3('n7', (-4, 3))
  p1 = Part(None, (i1, n7), i1 + n7)
  p2 = Part(None, (m7, i3), m7 * i3)
  p3 = Part(None, (m1, n2), m1 ** n2)
  parts = to_disjoint_parts([p1, p2, p3])
  def check_indices(parts, data) :
    return all(starmap(lambda p, d : renum_dummy(p.indices) == d, zip(parts, data)))
  def check_terms(parts, data) :
    return all(starmap(lambda p, d : renum_dummy(p.term) == d, zip(parts, data)))
  assert check_indices(parts, [(t1('_Dummy_1', (2, 5)), t3('_Dummy_4', (-3, 1))), (t1('_Dummy_1', (2, 5)), t3('_Dummy_2', (2, 3))), (t1('_Dummy_1', (2, 5)), t3('_Dummy_2', (-4, -4))), (t1('_Dummy_1', (0, 1)), t3('_Dummy_2', (-3, 1))), (t1('_Dummy_1', (6, 7)), t3('_Dummy_3', (-3, 1))), (t1('_Dummy_1', (2, 5)), t3('_Dummy_3', (-5, -5))), (t1('_Dummy_1', (6, 7)), t3('_Dummy_5', (2, 3))), (t1('_Dummy_1', (6, 7)), t3('_Dummy_4', (-4, -4))), (t1('_Dummy_1', (8, 10)), n7), (t1('_Dummy_1', (0, 1)), t3('_Dummy_4', (2, 3))), (t1('_Dummy_1', (0, 1)), t3('_Dummy_3', (-4, -4)))]) == True
  assert check_terms(parts, [t1('_Dummy_1', (2, 5)) + t3('_Dummy_4', (-3, 1)) + t1('_Dummy_1', (2, 5)) * t3('_Dummy_4', (-3, 1)) + t1('_Dummy_1', (2, 5))**t3('_Dummy_4', (-3, 1)), t1('_Dummy_1', (2, 5)) + t3('_Dummy_2', (2, 3)) + t1('_Dummy_1', (2, 5)) * t3('_Dummy_2', (2, 3)), t1('_Dummy_1', (2, 5)) + t3('_Dummy_2', (-4, -4)) + t1('_Dummy_1', (2, 5)) * t3('_Dummy_2', (-4, -4)), t1('_Dummy_1', (0, 1)) + t3('_Dummy_2', (-3, 1)) + t1('_Dummy_1', (0, 1))**t3('_Dummy_2', (-3, 1)), t1('_Dummy_1', (6, 7)) + t3('_Dummy_3', (-3, 1)) + t1('_Dummy_1', (6, 7))**t3('_Dummy_3', (-3, 1)), t1('_Dummy_1', (2, 5)) * t3('_Dummy_3', (-5, -5)), t1('_Dummy_1', (6, 7)) + t3('_Dummy_5', (2, 3)), t1('_Dummy_1', (6, 7)) + t3('_Dummy_4', (-4, -4)), t1('_Dummy_1', (8, 10)) + n7, t1('_Dummy_1', (0, 1)) + t3('_Dummy_4', (2, 3)), t1('_Dummy_1', (0, 1)) + t3('_Dummy_3', (-4, -4))]) == True

  s1 = ShiftedIdx(t1('s1', (0, 1)), 2)
  s2 = t1('s2', (2, 7))
  s3 = t1('s3', (1, 5))
  u1 = t1('u1', (0, 2))
  u2 = t1('u2', (1, 5))
  u3 = ShiftedIdx(t1('u3', (1, 8)), -1)
  p1 = Part(None, (s1, u1), s1 + u1)
  p2 = Part(None, (s3, u2), s3 * u2)
  p3 = Part(None, (s2, u3), s2**u3)
  parts = to_disjoint_parts([p1, p2, p3])
  assert check_indices(parts, [(t1('_Dummy_1', (2, 3)), t1('_Dummy_3', (1, 2))), (t1('_Dummy_1', (1, 1)), u2), (t1('_Dummy_1', (6, 7)), u3), (t1('_Dummy_1', (2, 3)), t1('_Dummy_3', (0, 0))), (t1('_Dummy_1', (2, 5)), t1('_Dummy_6', (6, 7))), (t1('_Dummy_31', (4, 5)), t1('_Dummy_1', (0, 0))), (t1('_Dummy_27', (4, 5)), t1('_Dummy_1', (1, 5))), (t1('_Dummy_1', (2, 3)), t1('_Dummy_5', (3, 5)))]) == True
  assert check_terms(parts, [t1('_Dummy_1', (2, 3)) + t1('_Dummy_3', (1, 2)) + t1('_Dummy_1', (2, 3)) * t1('_Dummy_3', (1, 2)) + t1('_Dummy_1', (2, 3))**t1('_Dummy_3', (1, 2)), t1('_Dummy_1', (1, 1)) * u2, t1('_Dummy_1', (6, 7))**u3, t1('_Dummy_1', (2, 3)) + t1('_Dummy_3', (0, 0)) + t1('_Dummy_1', (2, 3))**t1('_Dummy_3', (0, 0)), t1('_Dummy_1', (2, 5))**t1('_Dummy_6', (6, 7)), t1('_Dummy_31', (4, 5))**t1('_Dummy_1', (0, 0)), t1('_Dummy_27', (4, 5)) * t1('_Dummy_1', (1, 5)) + t1('_Dummy_27', (4, 5))**t1('_Dummy_1', (1, 5)), t1('_Dummy_1', (2, 3)) * t1('_Dummy_5', (3, 5)) + t1('_Dummy_1', (2, 3))**t1('_Dummy_5', (3, 5))]) == True

  p1 = Part(None, (m1, n1, k1), m1 + n1 + k1)
  p2 = Part(None, (m3, n4, k2), m3 * n4 * k2)
  parts = to_disjoint_parts([p1, p2])
  assert check_indices(parts, [(t1('_Dummy_1', (0, 3)), t3('_Dummy_3', (-3, -1)), t2('_Dummy_5', (4, 6))), (t1('_Dummy_1', (0, 3)), t3('_Dummy_3', (-3, -1)), t2('_Dummy_7', (7, 8))), (t1('_Dummy_1', (0, 3)), t3('_Dummy_3', (-3, -1)), t2('_Dummy_6', (2, 3))), (t1('_Dummy_1', (0, 3)), t3('_Dummy_4', (-5, -4)), k1), (t1('_Dummy_1', (4, 7)), n1, k1)]) == True
  assert check_terms(parts, [t1('_Dummy_1', (0, 3)) + t3('_Dummy_3', (-3, -1)) + t2('_Dummy_5', (4, 6)) + t1('_Dummy_1', (0, 3)) * t3('_Dummy_3', (-3, -1)) * t2('_Dummy_5', (4, 6)), t1('_Dummy_1', (0, 3)) + t3('_Dummy_3', (-3, -1)) + t2('_Dummy_7', (7, 8)), t1('_Dummy_1', (0, 3)) + t3('_Dummy_3', (-3, -1)) + t2('_Dummy_6', (2, 3)), t1('_Dummy_1', (0, 3)) + t3('_Dummy_4', (-5, -4)) + k1, t1('_Dummy_1', (4, 7)) + n1 + k1]) == True

  l1 = t1('l1')
  l2 = t1('l2')
  l3 = t1('l3', (1, 10))
  sl3 = ShiftedIdx(l3, -1)
  p1 = Part(None, (l1, l2), l1 + l2)
  p2 = Part(None, (l1, l1), l1**2)
  p3 = Part(None, (sl3, sl3), sl3**4)
  p4 = Part(None, (l3, sl3), sl3**6)
  parts = to_disjoint_parts([p1, p2, p3, p4])
  check_indices(parts, [(t1('_Dummy_1', (1, 10)), ShiftedIdx(t1('_Dummy_1', (1, 10)), -1)), (t1('_Dummy_1', (10, 10)), t1('_Dummy_1', (10, 10))), (t1('_Dummy_1', (2, 9)), t1('_Dummy_1', (0, 0))), (t1('_Dummy_1', (10, 10)), t1('_Dummy_4', (0, 8))), (t1('_Dummy_1', (1, 9)), t1('_Dummy_3', (1, 9))), (t1('_Dummy_1', (0, 9)), t1('_Dummy_1', (0, 9))), (t1('_Dummy_55', (1, 9)), t1('_Dummy_1', (10, 10))), (t1('_Dummy_1', (0, 0)), t1('_Dummy_3', (1, 10)))])
  check_terms(parts, [t1('_Dummy_1', (1, 10)) + ShiftedIdx(t1('_Dummy_1', (1, 10)), -1) + t1('_Dummy_1', (1, 10))**6, 2 * t1('_Dummy_1', (10, 10)) + t1('_Dummy_1', (10, 10))**2, t1('_Dummy_1', (2, 9)) + t1('_Dummy_1', (0, 0)), t1('_Dummy_1', (10, 10)) + t1('_Dummy_4', (0, 8)), t1('_Dummy_1', (1, 9)) + t1('_Dummy_3', (1, 9)), 2*t1('_Dummy_1', (0, 9)) + t1('_Dummy_1', (0, 9))**2 + t1('_Dummy_1', (0, 9))**4, t1('_Dummy_55', (1, 9)) + t1('_Dummy_1', (10, 10)), t1('_Dummy_1', (0, 0)) + t1('_Dummy_3', (1, 10))])
  pd = parts[0].get_pos((0,))
  assert pd.shift == -1
  pd = parts[1].get_pos((0,))
  assert pd.shift == 0
  pd = parts[2].get_pos((0,))
  assert type(pd) == _Full
  pd = parts[3].get_pos((0,))
  assert type(pd) == _Full
  pd = parts[4].get_pos((0,))
  assert pd.excludes == frozenset({0, -1})
  pd = parts[5].get_pos((0,))
  assert pd.shift == 0
  pd = parts[6].get_pos((0,))
  assert type(pd) == _Full
  pd = parts[7].get_pos((0,))
  assert type(pd) == _Full

  p1 = Part(None, (n1, m1), m1 + n1)
  p2 = Part(None, (n1, m4), m4 + n1)
  parts = to_disjoint_parts([p1, p2])
  assert len(parts) == 1 and renum_dummy(parts[0].indices) == (n1, t1('_Dummy_1', (0, 10))) and renum_dummy(parts[0].term) == t1('_Dummy_1', (0, 10)) + n1

  su1 = ShiftedIdx(u1, 1)
  su2 = ShiftedIdx(u1, 2)
  assert cmp_with_diag((i1, su1), (i1, su2)) == (0, 0, 33)
  assert cmp_with_diag((i1, su2), (i1, su1)) == (33, 0, 0)
  assert cmp_with_diag((n4,), (n5,)) == (0, 0, 6)
  assert cmp_with_diag((n5,), (n4,)) == (6, 0, 0)
  assert cmp_with_diag((n5,), (n5,)) == (0, 2, 0)
  assert cmp_with_diag((), ()) == (0, 1, 0)
  assert cmp_with_diag((i3, n5), (i3, n4)) == (54, 0, 0)
  assert cmp_with_diag((i3, n4), (i3, n5)) == (0, 0, 54)
  assert cmp_with_diag((i3, n5), (i3, n5)) == (0, 18, 0)
  assert cmp_with_diag((i1, u1), (i1, u2)) == (11, 22, 132)
  assert cmp_with_diag((i1, u1, m2), (i1, s1, m3)) == (308, 0, 1540)
  assert cmp_with_diag((i1, s1, m2), (i1, u1, m3)) == (1848, 0, 0)
  assert cmp_with_diag((i1, s1, m3), (i1, u1, m2)) == (1540, 0, 308)
  assert cmp_with_diag((i1, u1, m3), (i1, s1, m2)) == (0, 0, 1848)
  assert cmp_with_diag((m1, n1), (m2, n2)) == (162, 12, 1226)
  assert cmp_with_diag((m2, n2), (m1, n1)) == (1226, 12, 162)
  assert cmp_with_diag((m2, n2, u1), (m1, n1, u1)) == (3678, 36, 486)
  assert cmp_with_diag((m2, u1, n2), (m1, u1, n1)) == (3678, 36, 486)

  p = Part(None, (m0, n1, sm0), sm0**2)
  assert len(p) == 50
  assert renum_dummy(cxxcode(CodeBlock(*p.generate_loop([Assignment(Symbol('x'), p.term)])))).split('\n') == [
          'for (_Dummy_1 = 0; _Dummy_1 < 10; _Dummy_1 += 1) {',
          '   for (n1 = -5; n1 < 0; n1 += 1) {',
          '      x = std::pow(1 + _Dummy_1, 2);',
          '   };',
          '};']

  p = Part(None, (k2, m3, m5, sk2), m3)
  assert len(p) == 72
  assert renum_dummy(cxxcode(CodeBlock(*p.generate_loop([Assignment(Symbol('x'), p.term)])))).split('\n') == [
          'for (_Dummy_1 = 4; _Dummy_1 < 7; _Dummy_1 += 1) {',
          '   for (m3 = 0; m3 < 4; m3 += 1) {',
          '      for (m5 = 0; m5 < 6; m5 += 1) {',
          '         x = m3;',
          '      };',
          '   };',
          '};']
  assert renum_dummy(cxxcode(CodeBlock(*p.generate_loop([Assignment(Symbol('x'), p.term)], continue_cond = Symbol('x') < Symbol('y'))))).split('\n') == [
          'for (_Dummy_1 = 4; _Dummy_1 < 7; _Dummy_1 += 1) {',
          '   for (m3 = 0; m3 < 4; m3 += 1) {',
          '      for (m5 = 0; m5 < 6; m5 += 1) {',
          '         if (x < y) {',
          '            continue;',
          '         };',
          '         x = m3;',
          '      };',
          '   };',
          '};']
  p = p.set_pos(_Excl((1, 2), (m3, m5), {0}))
  assert len(p) == 68
  assert renum_dummy(cxxcode(CodeBlock(*p.generate_loop([Assignment(Symbol('x'), p.term)])))).split('\n') == [
          'for (_Dummy_1 = 4; _Dummy_1 < 7; _Dummy_1 += 1) {',
          '   for (m3 = 0; m3 < 4; m3 += 1) {',
          '      for (m5 = 0; m5 < 6; m5 += 1) {',
          '         if (m3 == m5) {',
          '            continue;',
          '         };',
          '         x = m3;',
          '      };',
          '   };',
          '};']
  assert renum_dummy(cxxcode(CodeBlock(*p.generate_loop([Assignment(Symbol('x'), p.term)], continue_cond = Symbol('x') < Symbol('y'))))).split('\n') == [
          'for (_Dummy_1 = 4; _Dummy_1 < 7; _Dummy_1 += 1) {',
          '   for (m3 = 0; m3 < 4; m3 += 1) {',
          '      for (m5 = 0; m5 < 6; m5 += 1) {',
          '         if (x < y) {',
          '            continue;',
          '         };',
          '         if (m3 == m5) {',
          '            continue;',
          '         };',
          '         x = m3;',
          '      };',
          '   };',
          '};']

  p = p.set_pos(_Excl((1, 2), (m3, m5), {-1, 0, 1}))
  assert len(p) == 61
  assert renum_dummy(cxxcode(CodeBlock(*p.generate_loop([Assignment(Symbol('x'), p.term)], continue_cond = Symbol('x') < Symbol('y'))))).split('\n') == [
          'for (_Dummy_1 = 4; _Dummy_1 < 7; _Dummy_1 += 1) {',
          '   for (m3 = 0; m3 < 4; m3 += 1) {',
          '      for (m5 = 0; m5 < 6; m5 += 1) {',
          '         if (x < y) {',
          '            continue;',
          '         };',
          '         if (m3 == -1 + m5 || m3 == 1 + m5 || m3 == m5) {',
          '            continue;',
          '         };',
          '         x = m3;',
          '      };',
          '   };',
          '};']


  print('ALL TESTS HAVE BEEN PASSED!!!')
