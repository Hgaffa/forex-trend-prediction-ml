import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from eda import EDA
from ta import TechnicalAnalysis
from sa import SentimentAnalysis
from fa import FundamentalAnalysis
from fs import FeatureSelection
from heatmap import Heatmap
from corr_plot import CorrelationPlot
from pca import PCAPlot
from models import Models

import seaborn as sns

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from stationary import Stationary

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tkinter import *


class MainApplication(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, *kwargs)

        self.df = None

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

        self.initialize_model_frames()

        self.initialize_feature_selection()

        self.initialize_visualizations()

        self.initialise_feature_dropdown()

        self.plotter = None

    def initialize_model_frames(self):

        #frame for val_curve
        self.val_curve_frame = tk.LabelFrame(root, text="Validation Curve")
        self.val_curve_frame.place(height = 300, width = 900, relx=0.53, rely=0.7)

        fig4 = plt.Figure()

        self.vc_canvas = FigureCanvasTkAgg(fig4,master=self.val_curve_frame)
        self.vc_toolbar = NavigationToolbar2Tk(self.vc_canvas, self.val_curve_frame)
        self.vc_toolbar.update()
        self.vc_plotter = self.vc_canvas.get_tk_widget()
        self.vc_plotter.pack(fill='both',anchor='center')

        #frame for stdout evaluation
        self.eval_frame = tk.LabelFrame(root, text="Model Results")
        self.eval_frame.place(height=120, width= 450, relx=0.53, rely=0.27)

        #frame for returns plot
        fig3 = plt.Figure()
        
        self.backtest_frame = tk.LabelFrame(root, text="Backtest Returns")
        self.backtest_frame.place(height = 300, width = 900, relx=0.53, rely=0.4)

        self.bt_canvas = FigureCanvasTkAgg(fig3,master=self.backtest_frame)
        self.bt_toolbar = NavigationToolbar2Tk(self.bt_canvas, self.backtest_frame)
        self.bt_toolbar.update()
        self.bt_plotter = self.bt_canvas.get_tk_widget()
        self.bt_plotter.pack(fill='both',anchor='center')

        #frame for confusion matrix
        self.cm_frame = tk.LabelFrame(root, text="Model Conufsion Matrix")
        self.cm_frame.place(height=400, width=400, relx=0.78)

        fig2 = plt.Figure()

        #for plotting confusion matrix
        self.cm_canvas = FigureCanvasTkAgg(fig2,master=self.cm_frame)
        self.cm_canvas.get_tk_widget().pack(fill='both')
        #frame for choosing classifier
        self.classifier_frame = tk.LabelFrame(root, text="Choose Classifier")
        self.classifier_frame.place(height=60, width=450, relx=0.53)

        #frame for choosing model parameters
        self.parameters = tk.LabelFrame(root, text="Set Model Parameters")
        self.parameters.place(height=200, width=450, relx=0.53, rely=0.07)

        self.variable_model_choice = tk.StringVar(root)
        self.variable_model_choice.set("KNN") #default value

        self.models_dropdown = OptionMenu(self.classifier_frame, self.variable_model_choice, *["KNN",'SVM','RF','ADB'], command= lambda event: self.set_model_params())
        self.models_dropdown.place(relx=0.14, rely=0.5, anchor='center')

        train_button = tk.Button(self.classifier_frame, text="Evaluate Model", command=lambda: self.train_model(),padx=10)
        train_button.place(rely=0.5, relx=0.45, anchor='center')

        backtest_button = tk.Button(self.classifier_frame, text="Backtest Model", command=lambda: self.backtest_model(),padx=10)
        backtest_button.place(rely=0.5, relx=0.8, anchor='center')

    def backtest_model(self):

        plt.clf()

        fig2, ax2 = plt.subplots(figsize=(8,2.5))
        ax2.clear()
        ax2.grid(True)
        ax2.set_title("Model Cumulative Returns")
        ax2.set_ylabel("Returns")
        ax2.set_xlabel("Date")

        ax2.plot(self.norm, label="Returns")
        ax2.plot(self.strategy, label="Strategy")
        ax2.legend()

        self.bt_canvas.figure = fig2

        self.bt_canvas.draw()

        #output model results financials
        sr_label = tk.Label(self.eval_frame, text="Sharpe's Ratio: {0:.2f}".format(self.sr.values[0]), padx=10)
        sr_label.grid(row = 1, column = 0, sticky = 'W', pady = 2) 

        eq_before_label = tk.Label(self.eval_frame, text="Equity before backtest: £1000.00", padx=10)
        eq_before_label.grid(row = 2, column = 0, sticky = 'W', pady = 2) 
        
        eq_after_label = tk.Label(self.eval_frame, text="Equity after backtest: £{0:.2f}".format(self.total.values[0]), padx=10)
        eq_after_label.grid(row = 3, column = 0, sticky = 'W', pady = 2) 

        tot_return_label = tk.Label(self.eval_frame, text="Total Return: {0:.2f}%".format((self.total.values[0] - 1000)/1000), padx=10)
        tot_return_label.grid(row = 4, column = 0, sticky = 'W', pady = 2) 

    def train_model(self):

        if self.variable_model_choice.get() == "KNN":

            clf, X_test, y_test, pred, self.sr, self.total, self.strategy, self.norm, training_sets, train_scores, val_scores = Models(root, self.df, self.old_returns.shift(-1)).get_knn(self.num_neighb_var.get(),  self.metric_knn.get(), self.alg_knn.get())

        elif self.variable_model_choice.get() == "RF":

            clf, X_test, y_test, pred, self.sr, self.total, self.strategy, self.norm, training_sets, train_scores, val_scores = Models(root, self.df, self.old_returns.shift(-1)).get_rf(int(self.num_est_rf.get()), int(self.mss_rf.get()))

        elif self.variable_model_choice.get() == "SVM":

            clf, X_test, y_test, pred, self.sr, self.total, self.strategy, self.norm, training_sets, train_scores, val_scores = Models(root, self.df, self.old_returns.shift(-1)).get_svm(self.kernel_svm.get(), float(self.c_svm.get()), float(self.gamma_svm.get()), self.cw_svm.get())

        plt.clf()

        fig, ax = plt.subplots(figsize=(4,4))
        ax.clear()
        ax.grid(False)

        plot_confusion_matrix(clf, X_test, y_test,                           
                                            display_labels=[-1,1],
                                            cmap=plt.cm.Blues, ax=ax)
        
        self.cm_canvas.figure = fig

        self.cm_canvas.draw()
        
        plt.clf()

        fig3, ax3 = plt.subplots(figsize=(8,2.5))
        ax3.clear()
        ax3.plot(training_sets, train_scores, c='gold', label='Training Error')
        ax3.plot(training_sets, val_scores, c='blue', label='Validation Error')

        ax3.set_title("Validation Curve")
        ax3.legend()

        self.vc_canvas.figure = fig3

        self.vc_canvas.draw()

    def set_model_params(self):

        #destroy old widget params in params frame
        for widget in self.parameters.winfo_children():
            widget.destroy()

        if self.variable_model_choice.get() == "KNN":

            self.knn()

        elif self.variable_model_choice.get() == "RF":

            self.rf()

        elif self.variable_model_choice.get() == "SVM":

            self.svm()

    def svm(self):

        #widgets and menu for kernels
        kernel_label_svm = tk.Label(self.parameters, text="Kernel:", padx=10)
        kernel_label_svm.grid(row = 1, column = 0, sticky = 'W', pady = 2) 

        self.kernel_svm = tk.StringVar(root)
        self.kernel_svm.set('rbf') #default value

        self.kernel_svm_menu = OptionMenu(self.parameters, self.kernel_svm, *['rbf'])
        self.kernel_svm_menu.grid(row = 1, column = 1, pady = 2) 

        #widgets and menu for C list
        c_label_svm = tk.Label(self.parameters, text="C:", padx=10)
        c_label_svm.grid(row = 2, column = 0, sticky = 'W', pady = 2) 

        gamma_exp = [-15,-13,-11,-9,-7,-5,-3,-1,1,3]
        c_exp = [-5,-3,-1,1,3,5,7,9,11,13,15]

        gamma_list = []
        c_list = []

        for i in gamma_exp:
            gamma_list.append(2**i)

        for i in c_exp:
            c_list.append(2**i)

        self.c_svm = tk.StringVar(root)
        self.c_svm.set(2) #default value

        self.c_svm_menu = OptionMenu(self.parameters, self.c_svm, *c_list)
        self.c_svm_menu.grid(row = 2, column = 1, pady = 2) 

        #widgets and menu for gamma
        gamma_label_svm = tk.Label(self.parameters, text="Gamma:", padx=10)
        gamma_label_svm.grid(row = 3, column = 0, sticky = 'W', pady = 2) 

        self.gamma_svm = tk.StringVar(root)
        self.gamma_svm.set(0.1) #default value

        self.gamma_svm_menu = OptionMenu(self.parameters, self.gamma_svm, *gamma_list)
        self.gamma_svm_menu.grid(row = 3, column = 1, pady = 2) 

        #widgets and menu for class weights
        cw_label_svm = tk.Label(self.parameters, text="Class Weight:", padx=10)
        cw_label_svm.grid(row = 4, column = 0, sticky = 'W', pady = 2) 

        self.cw_svm = tk.StringVar(root)
        self.cw_svm.set('balanced') #default value

        self.cw_svm_menu = OptionMenu(self.parameters, self.cw_svm, *['balanced', 'weighted'])
        self.cw_svm_menu.grid(row = 4, column = 1, pady = 2) 

    def rf(self):

        #widgets and menu for n_estimators
        numest_label_rf = tk.Label(self.parameters, text="N_estimators:", padx=10)
        numest_label_rf.grid(row = 1, column = 0, sticky = 'W', pady = 2) 

        self.num_est_rf = tk.StringVar(root)
        self.num_est_rf.set(100) #default value

        self.num_eft_rf_menu = OptionMenu(self.parameters, self.num_est_rf, *[100, 200, 400, 600, 800, 1000])
        self.num_eft_rf_menu.grid(row = 1, column = 1, pady = 2) 

        #widgets and menu for min_samples_split chosen parameter
        mss_label_rf = tk.Label(self.parameters, text="Min_samples_split:", padx=10)
        mss_label_rf.grid(row = 2, column = 0, sticky = 'W', pady = 2) 

        self.mss_rf = tk.StringVar(root)
        self.mss_rf.set(2) #default value

        self.mss_rf_menu = OptionMenu(self.parameters, self.mss_rf, *[2,3,4,5])
        self.mss_rf_menu.grid(row = 2, column = 1, pady = 2) 

    def knn(self):

        #widgets and menu for num neighbours parameter
        nn_label_knn = tk.Label(self.parameters, text="n_neighbours:", padx=10)
        nn_label_knn.grid(row = 0, column = 0, sticky = 'W', pady = 2) 

        num_neighbours = []
        for i in range(5, int(np.sqrt(len(self.df)))+1,2):

            num_neighbours.append(i)

        self.num_neighb_var = tk.StringVar(root)
        self.num_neighb_var.set(5) #default value

        self.num_neigh = OptionMenu(self.parameters, self.num_neighb_var, *num_neighbours)
        self.num_neigh.grid(row = 0, column = 1, pady = 2) 

        #widgets and menu for metric chosen parameter
        metric_label_knn = tk.Label(self.parameters, text="Metric:", padx=10)
        metric_label_knn.grid(row = 1, column = 0, sticky = 'W', pady = 2) 

        self.metric_knn = tk.StringVar(root)
        self.metric_knn.set('euclidean') #default value

        self.metric_knn_menu = OptionMenu(self.parameters, self.metric_knn, *['euclidean','manhattan','minkowski','chebyshev'])
        self.metric_knn_menu.grid(row = 1, column = 1, pady = 2) 

        #widgets and menu for alg chosen parameter
        alg_label_knn = tk.Label(self.parameters, text="Alg:", padx=10)
        alg_label_knn.grid(row = 2, column = 0, sticky = 'W', pady = 2) 

        self.alg_knn = tk.StringVar(root)
        self.alg_knn.set('auto') #default value

        self.alg_knn_menu = OptionMenu(self.parameters, self.alg_knn, *['auto', 'ball_tree', 'kd_tree', 'brute'])
        self.alg_knn_menu.grid(row = 2, column = 1, pady = 2) 

    def other(self):

        label = tk.Label(self.parameters, text="Nope")
        label.place(relx=0.2, rely=0.3)

    def initialize_visualizations(self):

        self.visualization_frame = tk.LabelFrame(root, text="Data Visualization")
        self.visualization_frame.place(height=150, width=200, rely=0.2, relx=0.41)

        correlation_button = tk.Button(self.visualization_frame, text="Correlation Plot", command=lambda: CorrelationPlot(root, self.df, self.old_returns.shift(-1)))
        correlation_button.place(relx=0.5, rely=0.2, anchor='center', width=150)

        heatmap_button = tk.Button(self.visualization_frame, text="Heatmap Plot", command=lambda: Heatmap(root, self.df))
        heatmap_button.place(relx=0.5, rely=0.47, anchor='center', width=150)

        pca_button = tk.Button(self.visualization_frame, text="PCA Plot", command=lambda: PCAPlot(root, self.df))
        pca_button.place(relx=0.5, rely=0.75, anchor='center', width=150)


    def initialize_labelling(self):

        self.labelling_frame = tk.LabelFrame(root, text="Generate Labels")
        self.labelling_frame.place(height=50, width=200, rely=0.25, relx=0.3)

        label_generation_button = tk.Button(self.labelling_frame, text="Label Data", command=lambda: self.generate_labels(),padx=10)
        label_generation_button.place(relx = 0.5, rely = 0.5, anchor='center')


    #function to generate labels via fixed horizon
    def generate_labels(self):

        self.og_df.index = pd.to_datetime(self.og_df.index)

        data = self.og_df[self.og_df.index.isin(self.df.index)]

        data['HL_Avg_Rolling'] = pd.DataFrame((data.High_MA + data.Low_MA)/2)

        data['HL_Avg'] = pd.DataFrame((data.High + data.Low)/2)

        returns = np.log(data.HL_Avg_Rolling) - np.log(data.HL_Avg_Rolling.shift(1))

        self.old_returns = np.log(data.Close) - np.log(data.Close.shift(1))

        self.labels = pd.DataFrame(returns.shift(-1).values, index=returns.index, columns=['Labels']).applymap(lambda x: 1 if x>= 0 else -1).astype(int)

        print(self.df.index)

        print(self.og_df.index)

        self.df['Labels'] = self.labels.values

        print(self.labels.value_counts())

        self.tree_update()

        self.feature_dropdown_update()


    #function to initialise frame for feature selection
    def initialize_feature_selection(self):

        self.feature_selection_frame = tk.LabelFrame(root, text="Feature Selection")
        self.feature_selection_frame.place(height=50, width=200, rely=0.3, relx=0.3)

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
        self.transform_frame.place(height=50, width=200, rely=0.2, relx=0.3)

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

        fig = plt.figure()

        self.canvas = FigureCanvasTkAgg(fig,master=self.close_plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.close_plot_frame)
        self.toolbar.update()
        self.plotter = self.canvas.get_tk_widget()
        self.plotter.pack(fill='both')

    def update_plot(self):

        #Plot chosen feature
        plt.rc('xtick',labelsize=6)
        plt.rc('ytick',labelsize=6)

        plt.clf()


        fig, ax = plt.subplots(figsize=(10,3.2))

        ax.plot(self.df[self.variable.get()], label=str(self.variable.get()))
        ax.grid(True)
        ax.legend()

        ax.set_title(self.variable.get())
        ax.set_ylabel(self.variable.get())

        self.canvas.figure = fig

        self.canvas.draw()

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
        filename = filedialog.askopenfilename(initialdir="/E:/OneDrive/Documents/University/Year 3/Project/financial-ml-demo/Data",
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
        
        #data['Adj_Close'] = data.Close.ewm(alpha=0.1).mean()

        data['High_MA'] = data.High.rolling(3).mean()

        data['Low_MA'] = data.Low.rolling(3).mean()

        #data['Returns'] = pd.DataFrame((data.High_MA+data.Low_MA/2).pct_change()).values

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