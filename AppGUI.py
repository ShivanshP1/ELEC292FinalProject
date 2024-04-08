import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.discriminant_analysis import StandardScaler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)


def classify(dataset_CSV):

    loaded_model = joblib.load('LRM.pkl')
    scaler = StandardScaler()
    clf = make_pipeline(StandardScaler(), loaded_model) 

    features_scaled = scaler.fit_transform(dataset_CSV)

    predictions = clf.predict(features_scaled)
    dataset_CSV['median'] = np.where(predictions == 1, 'Walking', 'Jumping')
    
    print(dataset_CSV['median'])


def Load_Graph():
    file_path = filedialog.askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    if file_path:
        # Process the selected file (you can replace this with your own logic)
        print("Selected file:", file_path)
        data = pd.read_csv(file_path)
    #------------------------------------------------------
    #Graphing
    fig = Figure(figsize = (5, 5)) 
    x_axis = data['Time (s)']
    y_axis = data['Absolute acceleration (m/s^2)']
    plt.xlabel('Time (s)')
    plt.ylabel('Absolute acceleration (m/s^2)') 
    plot1 = fig.add_subplot() 
    plot1.plot(x_axis,y_axis) 

    canvas = FigureCanvasTkAgg(fig, master = root)   
    canvas.draw() 
    canvas.get_tk_widget().pack() 

    toolbar = NavigationToolbar2Tk(canvas,root) 
    toolbar.update() 
    #------------------------------------------------------
    #Classify the data
    classify(data)

    #------------------------------------------------------

# Create the main Tkinter window
root = tk.Tk()
root.title("ELEC292FinalProject")
root.geometry("500x600") 
# Create an "Import CSV File" button
import_button = tk.Button(root, text="Import CSV File", command=Load_Graph)
import_button.pack(pady=20)
import_button.pack(padx=50)
# root.configure(background="black")

#-------------------------


text = Label(root, text="")
text.pack()

#------------------------
# Run the Tkinter event loop

root.mainloop()
