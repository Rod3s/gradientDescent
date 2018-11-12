from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import time

def sphereCost(x):
    y=np.sum(np.square(x))
    return y

def birdFcn(x):
    X=x[0]
    Y=x[1]
    scores = np.sin(X) * np.exp((1 - np.cos(Y))**2) + np.cos(Y) * np.exp((1 - np.sin(X))**2) + (X - Y)** 2;
    return scores    

def gradient(costFun,taps,deltaTap=1e-4):
    grad=np.empty(taps.shape)
    for i,tap in enumerate(taps):
        taps_temp=taps.copy()
        taps_temp[i]+=deltaTap/2
        y_plus=costFun(taps_temp)
        taps_temp[i]-=deltaTap
        y_minus=costFun(taps_temp)
        grad[i]=(y_plus-y_minus)/(deltaTap)
    return grad


tapLimits=(-4,4)

meshLen=50

x=np.linspace(tapLimits[0],tapLimits[1],meshLen).round(1)
y=np.linspace(tapLimits[0],tapLimits[1],meshLen).round(1)

X,Y=np.meshgrid(x,y)
Z=np.empty(X.shape)
count=0
for ix,valx in enumerate(x):
    for iy,valy in enumerate(y):
        count+=1
        Z[ix,iy]=sphereCost(np.array([valx,valy]))
  
fig,ax=plt.subplots(figsize=(10,7))      
cax=ax.contour(Y,X,Z)
fig.colorbar(cax,ax=ax)






mu=1e-3 #step size
taps=np.array([-0.0,-1.0])

iterLen=1000

tapsM=np.empty((iterLen+1,len(taps)))
print
tapsM[0,:]=taps
cost=np.empty(iterLen+1)
for i in range(iterLen):
    grad=gradient(sphereCost,tapsM[i,:],1e-5)
    taps-=grad*mu
    tapsM[i+1,:]=taps
    cost[i]=sphereCost(tapsM[i,:])


    
print(taps)

tapsM=tapsM.transpose()
ax.quiver(tapsM[0,:-1], tapsM[1,:-1], tapsM[0,1:]-tapsM[0,:-1], tapsM[1,1:]-tapsM[1,:-1],
          scale_units='xy', angles='xy', scale=1, color='k')



#
#
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#
#
## Plot the surface.
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)


plt.show()
#
#
