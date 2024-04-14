# %%
# A = B = 10cm, C = 6cm - said by the tutor
# thickness = 2mm - 3rd dim

# %%
A = 10
B = 10
C = 6

Start_red_temp = 20 # celcius, const
Start_blue_boundary_temp = 20 # celcius, const
Start_room_temp = 20 # ma byc 20, chwilowo # celcius, start temp of area between boundary and heater 
resolution = 1 # cm - co ile dzielimy matrixa (whole model is a matrix)

width = int(2 * A + B)
height = int(2 * B + A)

simulation_time = 60
dt = 0.01
dx = 0.1
dy = 0.1



# %%
class Material:
    def __init__(self):
        self.p = 2700
        self.Cw = 900
        self.K = 237

# %%
alumina = Material()

cooper = Material()
cooper.p = 8920
cooper.Cw = 380
cooper.K = 401

stainless_steel = Material()
stainless_steel.p = 7860
stainless_steel.Cw = 450
stainless_steel.K = 58

# %%
material = cooper

# %%
import numpy as np

# %%
TIMESTEPS = int(simulation_time // dt)
nodes = np.empty((width+1, height+1, TIMESTEPS))
nodes.fill(Start_room_temp)


# %%
def is_heater(i,j):
    mid_i = width / 2 
    mid_j = height / 2 
    
    if max(abs(i - mid_i), abs(j - mid_j)) <= C / 2:
        return True
    return False

# %%
def is_boundary(i, j):
    if (i == 0 or i == width) and (j >= B and j <= B+A):
        return True
    
    if (i == A or i == A+B) and (j <= B or j >= B+A):
        return True
    
    if (j == 0 or j == height) and (i >= A and i <= A+B):
        return True
    
    if (j == B or j == B+A) and (i <= A or i >= A+B):
        return True
    
    return False

def get_temp_for_boundary(i, j, t):
    if should_calculate(i, j-1):
        return nodes[i, j-1, t]
    if should_calculate(i, j+1):
        return nodes[i, j+1, t]
    if should_calculate(i-1, j):
        return nodes[i-1, j, t]
    if should_calculate(i+1, j):
        return nodes[i+1, j, t]
    
    if should_calculate(i-1, j-1):
        return nodes[i-1, j-1, t]
    if should_calculate(i-1, j+1):
        return nodes[i-1, j+1, t]
    if should_calculate(i+1, j-1):
        return nodes[i+1, j-1, t]
    
    return nodes[i+1, j+1, t]


# %%
def should_calculate(i, j):
    if (i <= A or i >= A+B) and (j >= B+A or j <= B):
        return False
    
    if (i-1 < 0 or i > width or j-1 < 0 or j > height):
        return False
    
    if is_boundary(i,j):
        return False

    if is_heater(i,j):
        return False
    return True

# %%
for i in range(width+1):
    for j in range(height+1):
        if is_heater(i, j):
            nodes[i, j, 0] = Start_red_temp
        elif is_boundary(i, j):
            nodes[i, j, 0] = Start_blue_boundary_temp

# %%
P = 100
B_2 = C*C / 10000
h = 0.002 # plate thickness

heating_time = 10 // dt

epsilon = 5

timesteps_need_to_stabilize = TIMESTEPS + 1

# %%
for t in range(1, TIMESTEPS):
    min_temp_t = nodes[0, 0, t]
    max_temp_t = nodes[0, 0, t]
    for i in range(width+1):
        for j in range(height+1):
            if is_heater(i, j) and t <= heating_time:
                nodes[i, j, t] = nodes[i, j, t-1] + (P*dt) / (material.Cw * B_2 * h * material.p)
            elif is_boundary(i, j):
                nodes[i, j, t] = get_temp_for_boundary(i, j, t-1)  
            elif should_calculate(i, j) or (is_heater(i, j) and t > heating_time):
                nodes[i, j, t] = nodes[i, j, t-1] + ( material.K / (material.Cw * material.p)) * \
                                ( (nodes[i+1, j, t-1] - 2*nodes[i,j,t-1] + nodes[i-1,j,t-1])/dx**2 + (nodes[i, j+1, t-1] - 2*nodes[i,j, t-1] + nodes[i,j-1, t-1])/dy**2 )
                
            if min_temp_t > nodes[i, j, t]:
                min_temp_t = nodes[i, j, t]

            if max_temp_t < nodes[i, j, t]:
                max_temp_t = nodes[i, j, t]

    if max_temp_t - min_temp_t < epsilon and timesteps_need_to_stabilize > TIMESTEPS and t > heating_time:
        timesteps_need_to_stabilize = t
            
print("Steps need to stabilize with heating:", timesteps_need_to_stabilize, f"this is {timesteps_need_to_stabilize * dt} seconds")  
print("Temperature:", nodes[15,15,TIMESTEPS-1])  

# %%
WIDTH = width + 1
HEIGHT = height + 1

# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

# Wyświetlenie pierwszej klatki danych temperatury
img = ax.imshow(nodes[:, :, 0], cmap='plasma', interpolation='nearest', vmin=nodes.min(), vmax=nodes.max())

# Definicja funkcji aktualizacji animacji
def update(frame):
    ax.set_title(f'Timestep: {frame}')
    img.set_data(nodes[:, :, frame])

# Utworzenie animacji
timesteps = list(range(0, TIMESTEPS, 4))
ani = FuncAnimation(fig, update, frames=timesteps, interval=40)

# Pokaż wykres
plt.colorbar(img)
plt.show()

# %%



