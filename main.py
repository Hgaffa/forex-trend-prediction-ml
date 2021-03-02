import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from eda import EDA
from ta import TechnicalAnalysis
from sa import SentimentAnalysis
from fa import FundamentalAnalysis
from fs import FeatureSelection

from stationary import Stationary

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tkinter import *


class MainApplication(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, *kwargs)
        self.parent = parent

        self.initialise_tree()

        self.initialise_open_file()

        self.eda()

        self.generate_features()
    
        #initialise frame for changing plot data (dropdown menu)
        self.choose_data = tk.LabelFrame(root, text="Choose Plot Data")
        self.choose_data.place(height=50, width=200, rely=0.3, relx=0)

        #initialise frame for plotting data
        self.close_plot_frame = tk.LabelFrame(root, text="Data Plot")
        self.close_plot_frame.place(height=400, width=1000, rely=0.35, relx=0)

        self.initialize_transform()

        self.initialize_labelling()

        self.initialize_feature_selection()

        self.initialise_feature_dropdown()

        self.plotter = None

    def initialize_labelling(self):

        self.labelling_frame = tk.LabelFrame(root, text="Generate Labels")
        self.labelling_frame.place(height=50, width=200, rely=0.25, relx=0.31)

        label_generation_button = tk.Button(self.labelling_frame, text="Label Data", command=lambda: self.generate_labels(),padx=10)
        label_generation_button.place(relx = 0.5, rely = 0.5, anchor='center')


    #function to generate labels via fixed horizon
    def generate_labels(self):

        data = self.og_df[self.og_df.index.isin(self.df.index)]

        data['HL_Avg_Rolling'] = pd.DataFrame((data.High_MA + data.Low_MA)/2)

        data['HL_Avg'] = pd.DataFrame((data.High + data.Low)/2)

        returns = np.log(data.HL_Avg) - np.log(data.HL_Avg.shift(1))

        self.labels = pd.DataFrame(returns.shift(-1).values, index=returns.index, columns=['Labels']).applymap(lambda x: 1 if x>= 0 else -1).astype(int)

        self.df['Labels'] = self.labels.values

        print(self.labels.value_counts())

        self.tree_update()

        self.feature_dropdown_update()

    #function to initialise frame for feature selection
    def initialize_feature_selection(self):

        self.feature_selection_frame = tk.LabelFrame(root, text="Feature Selection")
        self.feature_selection_frame.place(height=50, width=200, rely=0.3, relx=0.31)

        self.variable_fs = tk.StringVar(root)
        self.variable_fs.set(5) #default value

        self.num_features = OptionMenu(self.feature_selection_frame, self.variable_fs, *[5,8,12,15,17,20])
        self.num_features.place(relx=0.15, rely=0.45, anchor='center')

        fs_button = tk.Button(self.feature_selection_frame, text="Select Features", command=lambda: self.feature_selection(),padx=10)
        fs_button.place(rely=0.45, relx=0.7, anchor='center')

    def feature_selection(self):

        #make sure number of columns selected < num of columns in dataframe
        if int(self.variable_fs.get()) > len(self.df.columns):

            print("Bad Number of Feature to Select...")

        else:

            best_features = FeatureSelection(self.df).feature_selection(int(self.variable_fs.get()))

            if 'Labels' in self.df.columns:

                labels_index = self.df.columns.get_loc("Labels")

                best_features.append(labels_index)

            self.df = self.df[self.df.columns[best_features]]

            self.tree_update()

            self.feature_dropdown_update()

    def initialize_transform(self):

        #frame for adf test + output
        self.stationary_frame = tk.LabelFrame(root, text="ADF Test/Stationarity Output")
        self.stationary_frame.place(height=200, width=1000, rely=0.75, relx=0)

        #frame for transform button for stationarity
        self.transform_frame = tk.LabelFrame(root, text="Transform Data")
        self.transform_frame.place(height=50, width=200, rely=0.2, relx=0.31)

        button2 = tk.Button(self.transform_frame, text="Transfrom Data", command=lambda: self.data_transform(),padx=10)
        button2.place(relx = 0.5, rely= 0.5, anchor='center')

    #function to initialise dropdown menu in data plot frame and plot button
    def initialise_feature_dropdown(self):

        self.variable = tk.StringVar(root)
        self.variable.set("Close") #default value

        self.w = OptionMenu(self.choose_data, self.variable, *['Close'])
        self.w.place(relx=0.05, rely=0.1)

        plot_button = tk.Button(self.choose_data, text="Plot Feature", command=lambda: self.update_plot())
        plot_button.place(relx=0.55, rely=0.16)

    def update_plot(self):

        if self.plotter:

            self.plotter.destroy()
        #Plot chosen feature
        plt.rc('xtick',labelsize=6)
        plt.rc('ytick',labelsize=6)

        fig = Figure(figsize=(5,10))

        ax = fig.add_subplot(111)

        ax.plot(self.df[self.variable.get()])

        ax.set_title(self.variable.get(), fontsize=6)
        ax.set_ylabel(self.variable.get(), fontsize=6)
        ax.set_xlabel("Date", fontsize=6)

        canvas = FigureCanvasTkAgg(fig,master=self.close_plot_frame)
        self.plotter = canvas.get_tk_widget()
        self.plotter.pack(fill='both')
        canvas.draw()

        ax.clear()

    def feature_dropdown_update(self):

        #update dropdown menu items
        self.w['menu'].delete(0, 'end')
        self.variable.set("Close")
        for col in self.df.columns:
            self.w['menu'].add_command(label=col, command=tk._setit(self.variable, col))

    #function to initialise buttons for feature generations
    def generate_features(self):

        self.ta_frame = tk.LabelFrame(root, text="Feature Engineering")
        self.ta_frame.place(height=150, width=150, rely=0.2, relx=0.22)

        ta_button = tk.Button(self.ta_frame, text="Generate TA Features", command=lambda: self.update_tree_ta())
        ta_button.place(relx=0.5, rely=0.2, anchor='center')

        sa_button = tk.Button(self.ta_frame, text="Generate SA Features", command=lambda: self.update_tree_sa())
        sa_button.place(relx=0.5, rely=0.47, anchor='center')

        fa_button = tk.Button(self.ta_frame, text="Generate FA Features", command=lambda: self.update_tree_fa())
        fa_button.place(relx=0.5, rely=0.75, anchor='center')

    #function to generate TA features and propagate results
    def update_tree_ta(self):

        self.df = TechnicalAnalysis(self.df).ta()

        self.tree_update()

        self.feature_dropdown_update()

    #function to generate SA features and propagate results
    def update_tree_sa(self):

        self.df = SentimentAnalysis(self.df).sa()

        self.tree_update()

        self.feature_dropdown_update()

    #function to generate FA features and propagate results
    def update_tree_fa(self):

        self.df = FundamentalAnalysis(self.df).fa()

        self.tree_update()

        self.feature_dropdown_update()

    #function to initialize frame and button for detrending time series
    def stationarize(self):

        self.adf_text = StringVar()

        #initialise label to initial close series ADF test stats
        initial_adf = Stationary(self.df.dropna())
        
        check_adf = initial_adf.adf_test(self.df.Close)

        initial_adf_text = "The ADF Statistic of Close Price is: {:.2f} \n\n The p-Value is: {:.2f} \n\n Critical Values: \n ----------- \n".format(initial_adf.adf_stat, initial_adf.p_val)

        for key, val in initial_adf.crit_vals.items():
            
            temp = f'{key} : {val}'
            initial_adf_text += temp + "\n"

        initial_adf_text += "\n The Data is Now Stationary!" if initial_adf.p_val < initial_adf.THRESH else "\n The Data is Not Stationary!"

        self.adf_text.set(initial_adf_text)
        adf_label = tk.Label(self.stationary_frame, textvariable=self.adf_text).place(relx=0.5, rely=0.5, anchor='center')


    #function to transform data and make stationary
    def data_transform(self):

        data_st = pd.DataFrame(self.df.fillna(method='bfill')).applymap(lambda x: np.NAN if x in [np.inf, -np.inf] else x).dropna()

        stationary = Stationary(data_st)

        print(self.df)

        self.og_df = self.df

        #update df to be stationary df
        self.df = stationary.st()

        check_adf = stationary.adf_test(self.df.Close)

        adf_label_text = "The ADF Statistic of Close Price is: {:.2f} \n\n The p-Value is: {:.2f} \n\n Critical Values: \n ----------- \n".format(stationary.adf_stat, stationary.p_val)

        for key, val in stationary.crit_vals.items():
            
            temp = f'{key} : {val}'
            adf_label_text += temp + "\n"

        adf_label_text += "\n The Data is Now Stationary!" if stationary.p_val < stationary.THRESH else "\n The Data is Not Stationary!"


        self.adf_text.set(adf_label_text)

        #update tree view
        self.tree_update()

        return None

    #function to refresh values in price data tree view
    def tree_update(self):

        self.clear_data()

        self.tv1["column"] = list(self.df.columns)
        self.tv1["show"] = "headings"
        for column in self.tv1["columns"]:
            self.tv1.heading(column, text=column) # let the column heading = column name

        df_rows = self.df.to_numpy().tolist() # turns the dataframe into a list of lists
        for row in df_rows:
            self.tv1.insert("", "end", values=row)

    def eda(self):
        #Frame for performing EDA
        self.eda_frame = tk.LabelFrame(root, text="EDA")
        self.eda_frame.place(height=50, width=185,  rely=0.3, relx=0.11)

        button3 = tk.Button(self.eda_frame, text="Perform EDA", command=lambda: EDA(root, self.df[self.variable.get()]))
        button3.place(relx=0.5, rely=0.5, anchor='center')

    def initialise_open_file(self):
        # Frame for visualizing data
        self.file_frame = tk.LabelFrame(root, text="Open File")
        self.file_frame.place(height=100, width=400, rely=0.2, relx=0)

        # Buttons
        button1 = tk.Button(self.file_frame, text="Browse A File", command=lambda: self.File_dialog())
        button1.place(rely=0.65, relx=0.50)

        button2 = tk.Button(self.file_frame, text="Load File", command=lambda: self.Load_data())
        button2.place(rely=0.65, relx=0.30)

        # The file/file path text
        self.label_file = ttk.Label(self.file_frame, text="No File Selected")
        self.label_file.place(rely=0, relx=0)

    def create_window(self):
        window = tk.Toplevel(root)

    def initialise_tree(self):

         # Frame for TreeView
        self.frame1 = tk.LabelFrame(root, text="Price Data")
        self.frame1.place(height=200, width=1000, rely=0, relx=0)

        ## Treeview Widget
        self.tv1 = ttk.Treeview(self.frame1)
        self.tv1.place(relheight=1, relwidth=1) # set the height and width of the widget to 100% of its container (frame1).

        treescrolly = tk.Scrollbar(self.frame1, orient="vertical", command=self.tv1.yview) # command means update the yaxis view of the widget
        treescrollx = tk.Scrollbar(self.frame1, orient="horizontal", command=self.tv1.xview) # command means update the xaxis view of the widget

        self.tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set) # assign the scrollbars to the Treeview Widget

        treescrollx.pack(side="bottom", fill="x") # make the scrollbar fill the x axis of the Treeview widget
        treescrolly.pack(side="right", fill="y") # make the scrollbar fill the y axis of the Treeview widget

    def File_dialog(self):
        filename = filedialog.askopenfilename(initialdir="/E:/OneDrive/Documents/University/Year 3/Project/Notebooks/Data",
                                            title="Select A File",
                                            filetype=(("csv files", "*.csv"),("All Files", "*.*")))
        self.label_file["text"] = filename

    #Function to acquire and clean data
    def get_OHLC_data(self, filename):

        #Get dataset from Dukaskopy
        data = pd.read_csv(filename)

        data['Date'] = pd.to_datetime(data.Date, format="%d.%m.%Y %H:%M:%S.%f")

        data['Date'] = pd.DataFrame(data.Date).applymap(lambda x: x.date())

        data = data.set_index(data.Date).drop(columns=['Date'])
        
        #Remove data with volume of 0 as this corresponds to a weekend when no trading occurs
        data = data[data['Volume'] != 0]
        
        data['Adj_Close'] = data.Close.ewm(alpha=0.1).mean()

        data['High_MA'] = data.High.rolling(3).mean()

        data['Low_MA'] = data.Low.rolling(3).mean()

        data = data.dropna()
        
        return data

    def Load_data(self):
        file_path = self.label_file["text"]

        filename = r"{}".format(file_path)
        
        self.df = self.get_OHLC_data(filename)

        self.clear_data()

        self.og_df = self.df

        self.tv1["column"] = list(self.df.columns)
        self.tv1["show"] = "headings"
        for column in self.tv1["columns"]:
            self.tv1.heading(column, text=column) # let the column heading = column name

        df_rows = self.df.to_numpy().tolist() # turns the dataframe into a list of lists
        for row in df_rows:
            self.tv1.insert("", "end", values=row)

        #set inital adf values
        self.stationarize()

        self.feature_dropdown_update()

    def clear_data(self):
        self.tv1.delete(*self.tv1.get_children())

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Financial ML Project Demo")
    root.state("zoomed")
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.pack_propagate(False)
    root.mainloop()