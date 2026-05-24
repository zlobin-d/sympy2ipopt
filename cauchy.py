# vim: set fileencoding=utf8 :

from sympy import Matrix, Symbol


def _as_tuple(value, name):
  if value is None:
    return ()
  if isinstance(value, Matrix):
    return tuple(value)
  if isinstance(value, (tuple, list)):
    return tuple(value)
  raise TypeError(f'"{name}" should be a tuple, list or sympy Matrix.')


def _default_names(prefix, count):
  return tuple(f'{prefix}_{n}' for n in range(count))


class CauchySolution:
  '''Vector result of a Cauchy problem solution operator.

  The object is intentionally lightweight: it behaves like an immutable
  sequence of scalar SymPy expressions.  Each component is an application of a
  user function registered in :class:`sympy2ipopt.Nlp`.
  '''

  def __init__(self, problem, args, components):
    self.problem = problem
    self.args = tuple(args)
    self.components = tuple(components)

  def __len__(self):
    return len(self.components)

  def __iter__(self):
    return iter(self.components)

  def __getitem__(self, key):
    return self.components[key]

  def as_tuple(self):
    return self.components

  def as_matrix(self):
    return Matrix(self.components)

  def __repr__(self):
    return f'CauchySolution({self.components!r})'


class CauchyProblem:
  '''Description of a Cauchy problem used as an atomic NLP expression.

  This class is a high-level wrapper around ``Nlp.add_user_func``.  The
  solution operator of a vector ODE is represented component-wise:

  ``Phi(args) = (Phi_0(args), ..., Phi_{n-1}(args))``.

  The generated component functions are external C++ functions from the point
  of view of generated IpOpt code.  Their declarations are emitted by
  ``sympy2ipopt`` in ``<problem>_user_decls.h``.  A numerical backend must
  provide their definitions.

  Parameters
  ----------
  nlp:
      :class:`sympy2ipopt.Nlp` instance where user functions are registered.
  name:
      Prefix for generated function names.
  time:
      SymPy symbol denoting the independent variable.
  state:
      State variables of the ODE.
  rhs:
      Right-hand side expressions.  Must have the same length as ``state``.
  parameters:
      Finite-dimensional parameters of the ODE.
  controls:
      Finite-dimensional control parameters on the integration interval.
  include_time_bounds:
      If true, solution functions have ``t0`` and ``t1`` as their first two
      arguments.
  register_second_derivatives:
      If true, first derivative functions receive gradients made of second
      derivative functions.  This is useful because ``sympy2ipopt`` generates
      the exact Hessian of the Lagrangian.
  '''

  def __init__(self, nlp, name, *, time, state, rhs, parameters=(),
               controls=(), include_time_bounds=True,
               register_second_derivatives=True):
    self.nlp = nlp
    self.name = str(name)
    self.time = time
    self.state = _as_tuple(state, 'state')
    self.rhs = _as_tuple(rhs, 'rhs')
    self.parameters = _as_tuple(parameters, 'parameters')
    self.controls = _as_tuple(controls, 'controls')
    self.include_time_bounds = bool(include_time_bounds)
    self.register_second_derivatives = bool(register_second_derivatives)

    if not self.state:
      raise ValueError('"state" should contain at least one variable.')
    if len(self.rhs) != len(self.state):
      raise ValueError('"rhs" length should be equal to "state" length.')

    self.arg_names = self._make_arg_names()
    self.nargs = len(self.arg_names)
    if self.nargs <= 0:
      raise ValueError('Solution operator should have at least one argument.')

    self.second_derivative_functions = ()
    self.first_derivative_functions = ()
    self.solution_functions = ()
    self._register_functions()

  @property
  def dimension(self):
    return len(self.state)

  @property
  def argument_count(self):
    return self.nargs

  def _make_arg_names(self):
    names = []
    if self.include_time_bounds:
      names.extend(('t0', 't1'))
    names.extend(_default_names('x0', len(self.state)))
    names.extend(str(p) for p in self.parameters)
    names.extend(str(u) for u in self.controls)
    return tuple(names)

  def _register_functions(self):
    if self.register_second_derivatives:
      d2 = []
      for comp in range(self.dimension):
        comp_d2 = []
        for first in range(self.nargs):
          row = []
          for second in range(self.nargs):
            fname = f'{self.name}_d2_{comp}_{first}_{second}'
            row.append(self.nlp.add_user_func(fname, self.nargs))
          comp_d2.append(tuple(row))
        d2.append(tuple(comp_d2))
      self.second_derivative_functions = tuple(d2)

    d1 = []
    for comp in range(self.dimension):
      row = []
      for arg in range(self.nargs):
        fname = f'{self.name}_d1_{comp}_{arg}'
        grad = (self.second_derivative_functions[comp][arg]
                if self.register_second_derivatives else None)
        row.append(self.nlp.add_user_func(fname, self.nargs, grad=grad))
      d1.append(tuple(row))
    self.first_derivative_functions = tuple(d1)

    funcs = []
    for comp in range(self.dimension):
      fname = f'{self.name}_{comp}'
      funcs.append(self.nlp.add_user_func(
        fname,
        self.nargs,
        grad=self.first_derivative_functions[comp],
      ))
    self.solution_functions = tuple(funcs)

  def arguments(self, *, t0=None, t1=None, initial_state=None,
                parameters=None, controls=None):
    '''Build the ordered argument tuple for the solution operator.'''

    initial_state = _as_tuple(initial_state, 'initial_state')
    parameters = self.parameters if parameters is None else _as_tuple(parameters, 'parameters')
    controls = self.controls if controls is None else _as_tuple(controls, 'controls')

    if len(initial_state) != len(self.state):
      raise ValueError('"initial_state" length should be equal to state dimension.')
    if len(parameters) != len(self.parameters):
      raise ValueError('"parameters" length should match the problem parameters.')
    if len(controls) != len(self.controls):
      raise ValueError('"controls" length should match the problem controls.')

    args = []
    if self.include_time_bounds:
      if t0 is None or t1 is None:
        raise ValueError('"t0" and "t1" are required for this CauchyProblem.')
      args.extend((t0, t1))
    args.extend(initial_state)
    args.extend(parameters)
    args.extend(controls)
    return tuple(args)

  def solution(self, *, t0=None, t1=None, initial_state,
               parameters=None, controls=None):
    '''Return component expressions for ``x(t1)``.'''

    args = self.arguments(
      t0=t0,
      t1=t1,
      initial_state=initial_state,
      parameters=parameters,
      controls=controls,
    )
    components = tuple(func(*args) for func in self.solution_functions)
    return CauchySolution(self, args, components)

  def first_derivative(self, component, argument, *args):
    '''Return expression for d Phi_component / d arg_argument.

    ``args`` should be ordered in the same way as :meth:`arguments`.
    '''

    return self.first_derivative_functions[component][argument](*args)

  def second_derivative(self, component, first_argument, second_argument, *args):
    '''Return expression for a second partial derivative of a component.'''

    if not self.register_second_derivatives:
      raise RuntimeError('Second derivative functions were not registered.')
    return self.second_derivative_functions[component][first_argument][second_argument](*args)

  def rhs_jacobian_state(self):
    '''Jacobian of the ODE right-hand side with respect to state variables.'''

    return Matrix(self.rhs).jacobian(Matrix(self.state))

  def rhs_jacobian_parameters(self):
    '''Jacobian of the ODE right-hand side with respect to parameters.'''

    return Matrix(self.rhs).jacobian(Matrix(self.parameters)) if self.parameters else Matrix.zeros(self.dimension, 0)

  def rhs_jacobian_controls(self):
    '''Jacobian of the ODE right-hand side with respect to controls.'''

    return Matrix(self.rhs).jacobian(Matrix(self.controls)) if self.controls else Matrix.zeros(self.dimension, 0)

  def variational_rhs(self, *, state_direction=None, parameter_direction=None,
                      control_direction=None):
    '''Right-hand side of the direct variational equation.

    If directions are not provided, symbolic placeholders are created.  The
    returned expression is

    ``f_x eta + f_p q + f_u v``.
    '''

    eta = (_as_tuple(state_direction, 'state_direction')
           if state_direction is not None
           else tuple(Symbol(f'eta_{n}', real=True) for n in range(self.dimension)))
    q = (_as_tuple(parameter_direction, 'parameter_direction')
         if parameter_direction is not None
         else tuple(Symbol(f'q_{n}', real=True) for n in range(len(self.parameters))))
    v = (_as_tuple(control_direction, 'control_direction')
         if control_direction is not None
         else tuple(Symbol(f'v_{n}', real=True) for n in range(len(self.controls))))

    if len(eta) != self.dimension:
      raise ValueError('"state_direction" length should be equal to state dimension.')
    if len(q) != len(self.parameters):
      raise ValueError('"parameter_direction" length should match parameters.')
    if len(v) != len(self.controls):
      raise ValueError('"control_direction" length should match controls.')

    rhs = self.rhs_jacobian_state() * Matrix(eta)
    if self.parameters:
      rhs += self.rhs_jacobian_parameters() * Matrix(q)
    if self.controls:
      rhs += self.rhs_jacobian_controls() * Matrix(v)
    return rhs

  def initial_state_sensitivity_problem(self, name=None):
    '''Build an augmented Cauchy problem for ``d x(t) / d x(t0)``.

    The augmented state consists of the original state ``x`` and the flattened
    sensitivity matrix ``S``:

    ``y = (x, vec(S))``.

    For ``x' = f(t, x, p, u)`` the sensitivity matrix with respect to the
    initial state satisfies

    ``S' = f_x(t, x, p, u) * S,    S(t0) = I``.

    The returned object is another :class:`CauchyProblem`.  Its solution
    operator returns both ``x(t1)`` and all components of ``S(t1)``.  The
    entries of ``S(t1)`` are exactly the partial derivatives of the solution
    operator with respect to the initial state.
    '''

    n = self.dimension
    name = name or f'{self.name}_x0_sensitivity'

    sens_state = tuple(
      Symbol(f'{self.name}_S_{row}_{col}', real=True)
      for row in range(n)
      for col in range(n)
    )
    sens_matrix = Matrix(n, n, sens_state)
    sens_rhs = self.rhs_jacobian_state() * sens_matrix

    return CauchyProblem(
      self.nlp,
      name,
      time=self.time,
      state=self.state + sens_state,
      rhs=self.rhs + tuple(sens_rhs),
      parameters=self.parameters,
      controls=self.controls,
      include_time_bounds=self.include_time_bounds,
      register_second_derivatives=self.register_second_derivatives,
    )

  def initial_state_sensitivity_initial_value(self, initial_state=None):
    '''Initial value for the augmented ``d x(t) / d x(t0)`` problem.

    It is ``(x0, vec(I))`` and can be passed as ``initial_state`` to the
    problem returned by :meth:`initial_state_sensitivity_problem`.
    '''

    initial_state = (self.state if initial_state is None
                     else _as_tuple(initial_state, 'initial_state'))
    if len(initial_state) != self.dimension:
      raise ValueError('"initial_state" length should be equal to state dimension.')
    identity = tuple(Matrix.eye(self.dimension))
    return tuple(initial_state) + identity

  def split_initial_state_sensitivity_solution(self, solution):
    '''Split augmented solution into ``x(t1)`` and ``d x(t1) / d x(t0)``.

    Parameters
    ----------
    solution:
        A :class:`CauchySolution` returned by the augmented problem created by
        :meth:`initial_state_sensitivity_problem`.

    Returns
    -------
    tuple
        ``(state_solution, sensitivity_matrix)``.
    '''

    components = tuple(solution)
    n = self.dimension
    if len(components) != n + n * n:
      raise ValueError('Unexpected augmented solution dimension.')
    return components[:n], Matrix(n, n, components[n:])

  def adjoint_rhs(self, *, adjoint=None):
    '''Right-hand side for the backward adjoint equation.

    For ``x' = f(t, x, p, u)`` the convention is
    ``-lambda' = f_x.T * lambda``.  This method returns ``f_x.T * lambda``.
    '''

    lam = (_as_tuple(adjoint, 'adjoint')
           if adjoint is not None
           else tuple(Symbol(f'lambda_{n}', real=True) for n in range(self.dimension)))
    if len(lam) != self.dimension:
      raise ValueError('"adjoint" length should be equal to state dimension.')
    return self.rhs_jacobian_state().T * Matrix(lam)

  def parameter_quadrature_rhs(self, *, adjoint=None):
    '''Integrand for adjoint parameter gradients: ``f_p.T * lambda``.'''

    if not self.parameters:
      return Matrix.zeros(0, 1)
    lam = (_as_tuple(adjoint, 'adjoint')
           if adjoint is not None
           else tuple(Symbol(f'lambda_{n}', real=True) for n in range(self.dimension)))
    if len(lam) != self.dimension:
      raise ValueError('"adjoint" length should be equal to state dimension.')
    return self.rhs_jacobian_parameters().T * Matrix(lam)

  def control_quadrature_rhs(self, *, adjoint=None):
    '''Integrand for adjoint control-parameter gradients: ``f_u.T * lambda``.'''

    if not self.controls:
      return Matrix.zeros(0, 1)
    lam = (_as_tuple(adjoint, 'adjoint')
           if adjoint is not None
           else tuple(Symbol(f'lambda_{n}', real=True) for n in range(self.dimension)))
    if len(lam) != self.dimension:
      raise ValueError('"adjoint" length should be equal to state dimension.')
    return self.rhs_jacobian_controls().T * Matrix(lam)


__all__ = ['CauchyProblem', 'CauchySolution']
