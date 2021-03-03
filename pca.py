import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

import seaborn as sns

from sklearn.decomposition import PCA 

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tkinter import *


class PCAPlot(Toplevel): 
      
    def __init__(self, master = None, data = None): 
          
        super().__init__(master = master) 

        self.geometry("800x700")

        self.plotter=None

        #Frame for pca plot
        self.pca_frame = tk.LabelFrame(self, text="Correlation with future returns")
        self.pca_frame.place(height=700, width=800, rely=0, relx=0)

        self.df = data

        self.plot_pca()

    def plot_pca(self):

        if self.plotter:

            self.plotter.destroy()

        fig = Figure(figsize=(20, 17))
        ax = fig.subplots()

        X_copy = self.df.copy().drop(columns=['Labels'])

        scaler = StandardScaler()

        X_copy_scaled = scaler.fit_transform(X_copy)

        pca = PCA(n_components=2)

        pca.fit(X_copy_scaled)

        X_pca = pca.transform(X_copy_scaled)

        print("Old Shape: ", self.df.shape)

        print("New Shape: ", X_pca.shape)

        fig, ax = plt.subplots(figsize=(15,10))

        ax.scatter(X_pca[:,0], X_pca[:,1],c=self.df['Labels'])

        canvas = FigureCanvasTkAgg(fig,master=self.pca_frame)
        self.plotter = canvas.get_tk_widget()
        self.plotter.pack(fill='both')
        canvas.draw()