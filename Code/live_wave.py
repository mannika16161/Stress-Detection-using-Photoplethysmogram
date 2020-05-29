 # --- Imports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as Tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import pandas as pd
import cv2
# --- End of imports

fig = plt.Figure()


data = pd.read_csv('data.csv')
x = data['x_value']
y = data['total_1']

def animate_thread(i):
    threading.Thread(target=animate, args=(i,)).start()

def animate(i):

	global fig, x, y, line , ax
	data = pd.read_csv('data.csv')
	x = data['x_value']
	y = data['total_1']
	ax.cla()
	ax.plot(x, y, label='Channel 1')
	ax.legend(loc='upper left')
	ax.tight_layout()
	# return line


root = Tk.Tk()

label = Tk.Label(root, text="Graph")
label.grid(column=0, row=0)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(column=0, row=1)
endButton = Tk.Button(root, text='End', width=25, command=root.destroy) 
endButton.grid(column=0,row=2)


ax = fig.add_subplot(111)
ani = animation.FuncAnimation(fig, animate_thread, interval=1000, blit=False)

Tk.mainloop()