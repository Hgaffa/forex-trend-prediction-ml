import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

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
        self.pca_frame = tk.LabelFrame(self, text="PCA Plot of Top 3 Principal Components")
        self.pca_frame.place(height=700, width=800, rely=0, relx=0)

        self.df = data

        self.plot_pca()

    def plot_pca(self):

        if self.plotter:

            self.plotter.destroy()

        fig = Figure(figsize=(20, 17))

        canvas = FigureCanvasTkAgg(fig,master=self.pca_frame)
        toolbar = NavigationToolbar2Tk(canvas, self.pca_frame)
        toolbar.update()
        self.plotter = canvas.get_tk_widget()
        self.plotter.pack(fill='both')

        ax = fig.subplots()

        X_copy = self.df.copy().drop(columns=['Labels'])

        scaler = StandardScaler()

        X_copy_scaled = scaler.fit_transform(X_copy)

        pca = PCA(n_components=3)

        pca.fit(X_copy_scaled)

        X_pca=pd.DataFrame(pca.transform(X_copy_scaled), columns=['PCA%i' % i for i in range(3)])

        print("Old Shape: ", self.df.shape)

        print("New Shape: ", X_pca.shape)

        ax = fig.add_subplot(111, projection='3d')

        plot_colours = []

        for i in self.df['Labels'].astype(int):

            if i == 1:

                plot_colours.append('blue')
            
            else:

                plot_colours.append('yellow')

        ax.scatter(X_pca['PCA0'], X_pca['PCA1'], X_pca['PCA2'], c=plot_colours, cmap="Set2_r", s=60)

        import matplotlib.patches as mpatches

        legend_dict = { 'Uptrend': 'blue', 'Downtrend': 'yellow'}
        patch_list = []
        for key in legend_dict:
            data_key = mpatches.Patch(color=legend_dict[key], label=key)
            patch_list.append(data_key)
        ax.legend(handles=patch_list)
        
        # make simple, bare axis lines through space:
        xAxisLine = ((min(X_pca['PCA0']), max(X_pca['PCA0'])), (0, 0), (0,0))
        ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
        yAxisLine = ((0, 0), (min(X_pca['PCA1']), max(X_pca['PCA1'])), (0,0))
        ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
        zAxisLine = ((0, 0), (0,0), (min(X_pca['PCA2']), max(X_pca['PCA2'])))
        ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
        
        # label the axes
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA on the iris data set")
        