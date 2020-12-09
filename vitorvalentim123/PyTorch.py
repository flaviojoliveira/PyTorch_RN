import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
ages = np.random.randint(low=15, high=70, size=40)

ages

labels = []
for age in ages:
    if age < 30:
        labels.append(0)
    else:
        labels.append(1)
        
for i in range(0, 3):
    r = np.random.randint(0, len(labels) - 1)
    if labels[r] == 0:
        labels[r] = 1
    else:
        labels[r] = 0

plt.scatter(ages, labels, color="red")
plt.show()

import numpy as np
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(ages.reshape(-1, 1), labels)

m = model.coef_[0]
b = model.intercept_

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
 
axis = plt.axes(xlim =(0, 2),  
                ylim =(-0.1, 2))
 
line, = axis.plot([], [], lw = 3)  

def init():  
    line.set_data([], [])  
    return line,  

def animate(i):
    m_copy = i * 0.01
    plt.title('m = ' + str(m_copy))
    x = np.arange(0.0, 10.0, 0.1)
    y = m_copy * x + b
    line.set_data(x, y)  

    return line,

ani = FuncAnimation(fig, animate, init_func = init,  
                    frames = 200,  
                    interval = 20,  
                    blit = True)

ani.save('m.mp4', writer = 'ffmpeg', fps = 30)

from IPython.display import HTML

HTML("""
<div align="middle">
<video width="80%" controls>
      <source src="m.mp4" type="video/mp4">
</video></div>""")

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
  
axis = plt.axes(xlim =(0, 2),  
                ylim =(-0.1, 2))

line, = axis.plot([], [], lw = 3)  
  
def init():  
    line.set_data([], [])  
    return line,  

def animate(i):
    b_copy = i * 0.01
    plt.title('b = ' + str(b_copy))
    x = np.arange(0.0, 10.0, 0.1)
    y = m * x + b_copy
    line.set_data(x, y)  

    return line,

ani = FuncAnimation(fig, animate, init_func = init,  
                    frames = 200,  
                    interval = 20,  
                    blit = True)

ani.save('b.mp4',  
          writer = 'ffmpeg', fps = 30)

from IPython.display import HTML

HTML("""
<div align="middle">
<video width="80%" controls>
      <source src="b.mp4" type="video/mp4">
</video></div>""")

limiar_idade = (0.5 - b) / m
print(limiar_idade)

plt.plot(ages, ages * m + b, color = 'blue')
plt.plot([limiar_idade, limiar_idade], [0, 0.5], '--', color = 'green')
plt.scatter(ages, labels, color="red")
plt.show()

import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)

plt.plot(x, sig)
plt.show()

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(ages.reshape(-1, 1), labels)

m = model.coef_[0][0]
b = model.intercept_[0]

x = np.arange(0, 70, 0.1)
sig = sigmoid(m*x + b)

limiar_idade = 0 - (b / m)
print(limiar_idade)

plt.scatter(ages, labels, color="red")
plt.plot([limiar_idade, limiar_idade], [0, 0.5], '--', color = 'green')
plt.plot(x, sig)
plt.show()

