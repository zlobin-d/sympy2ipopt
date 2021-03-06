################################
sympy2ipopt
################################

sympy2ipopt является SymPy-интерфейсом для IpOpt.

SymPy -- система компьютерной алгебры (символьные вычисления), представляет из себя библиотеку Python с открытым исходным кодом.

IpOpt -- реализация численного метода внутренней точки для решения задач нелинейного программирования (НЛП) большой размерности,
представляет из себя библиотеку С++ с открытым исходным кодом.

sympy2ipopt предоставляет:

#. средства для описания задачи НЛП в постановке, используемой IpOpt, при помощи символьных выражений;
#. средства для аналитического расчета градинта целевого функционала,
   разреженных якобиана и гессиана лагранжиана задачи НЛП при помощи символьного дифференцирования;
#. средства для генерации кода на С++ в соответствии с API IpOpt;
   сгенерированная программа содержит математические выражения с использованием <cmath>,
   циклы, условные операторы и не использует динамическую работу с памятью, сложные структуры данных.
   
Назначение sympy2ipopt -- сгенерировать программу, способную численно решать задачу НЛП в реальном времени (например,
в рамках цикла управления) и подходящую для встраеваемых систем.
При этом сама кодогенерация может занимать достаточно продолжительное время.

Поддержка задач большой размерности обеспечивается за счет работы с индексированными переменными в рамках символьных вычислений.
Предполагается, что ограничения задачи НЛП и целевой функционал описаны при помощи относительно небольших формул,
использующих переменные с индексами, аналогично тому, как это делается при записи на бумаге разностных схем для приближения производных.
Например, простое выражение
:math:`\\frac{f_i - f_{i - 1}}{T} < v_{max}, i = \\overline{1,10000}`
задает целое семейство ограничений, размер которого равен длине диапазона индекса :math:`i`.
Изначально sympy2ipopt создавался для решения задач 'trajectory optimization',
большая размерность которых обусловлена работой с дискретизацией функций от времени.

API sympy2ipopt, требования к символьным переменным и выражениям описаны в документации классов Nlp, IdxType, ShiftedIdx.

