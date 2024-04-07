from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def import_file():
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Process the selected file (you can replace this with your own logic)
        print("Selected file:", file_path)
        data = pd.read_csv(file_path)
    fig = Figure(figsize = (5, 5), dpi = 100) 
    # list of squares 
    x_axis = data['Time (s)']
    y_axis = data['Absolute acceleration (m/s^2)']
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute acceleration (m/s^2)') 
    plot1 = fig.add_subplot() 
    plot1.plot(x_axis,y_axis) 
    canvas = FigureCanvasTkAgg(fig, master = root)   
    canvas.draw() 
  
    # placing the canvas on the Tkinter window 
    canvas.get_tk_widget().pack() 
  
    # # creating the Matplotlib toolbar 
    # toolbar = NavigationToolbar2Tk(canvas,root) 
    # toolbar.update() 
 

# Create the main Tkinter window
root = tk.Tk()
root.title("Plotting a graph from CSV")
root.geometry("500x600") 
# Create an "Import CSV File" button
import_button = tk.Button(root, text="Import CSV File", command=import_file)
import_button.pack(pady=20)
import_button.pack(padx=100)
# root.configure(background="black")

#-------------------------

text = Label(root, text="")
text.pack()

#------------------------
# Run the Tkinter event loop
root.mainloop()
