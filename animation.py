import pylab
pylab.ion()
N = 10000
t = pylab.linspace(0,10*pylab.pi,N)
sin = pylab.sin(t)+pylab.randn(N)*0.1
fig = pylab.figure()
l = pylab.plot(t[:1],sin[:1])[0]
pylab.ylim(-1,1)
for i in range(2,N):
    l.set_data(t[:i],sin[:i])
    pylab.xlim(0,t[i])
    
    pylab.draw()
    pylab.pause(0.01)



