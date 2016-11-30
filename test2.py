from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

x = T.matrix('x')
y = T.matrix('y')
z = function([x,y],T.exp(T.dot(x,y)))
rng = numpy.random.RandomState(22)
x = numpy.asarray([rng.rand(50*30)], config.floatX)
y = numpy.asarray([rng.rand(500)], config.floatX).T
print(z.maker.fgraph.toposort())
t0 = time.time()
print(z([[1],[2]],[[1,2]]))
for i in range(10000):
    z(y,x)
t1 = time.time()
print(t1-t0)

t3 = time.time()
for j in range(10000):
    numpy.exp(numpy.dot(y,x))
t4 = time.time()
print(t4-t3)