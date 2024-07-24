import numpy as np
import matplotlib.pyplot as plt

def Gaussion(x,dx):
    return 4 * np.pi * (3/(2*np.pi))**(3/2)*np.exp(-1.5*x**2)*x**2

dx = 0.05
x = np.arange(0,3,0.05)
y = Gaussion(x,dx) 

fig,ax = plt.subplots(figsize=(4.6,4))
ax.plot(x,y)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()