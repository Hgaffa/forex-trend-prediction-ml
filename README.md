# Forex ML Trend Prediction
The Forex market is the largest and most liquid market in the world, in which currency pairs are traded with the hopes of benefitting from the change in its exchange rate, this provides an incentive for the creation of algorithms to benefit from its volatility. With the ever-increasing use of machine learning, the question stands whether trends within the forex market can be predicted using machine learning.

This repository outlines the research exploring applications of machine learning and data analytics for forex trend classification. The full report/dissertation can be found in the `project.pdf` file.
## 1 Jupyter Notebooks:
A series of Jupyter Notebooks have been compiled and condensed to provide an overview of the research surrounding this project. The notebooks are named and well documented to guide through the theory and design choices surrounding EDA, Data Processing, Model Training & Evaluation etc.

It is important to note that this research follows an iterative ideology, synonymous to the iterative nature of data analysis. As more information is found about the data and model performance, further tuning and analysis can be performed. The following diagram outlines the workflow followed throughout the project and is identical in some areas to most supervised learning problem pipelines:

![image](https://user-images.githubusercontent.com/43887682/164040897-59eefa27-7826-4e39-935e-777c87bebcc4.png)

Details of each stage of the research can be found within the provided notebooks. To access the notebooks there are two options after cloning: 
- Open them in an editor such as VScode and install supporting packages for local jupyter servers
- Spin up a local server manually via the `python jupyter notebook` command. 

## 2 Machine Learning Dashboard UI:

The second part of this repo includes a tangible dashboard/UI that holistically demonstrates the theory explored within the research notebooks within a pipeline for prediction. To access the system, simply enter the appropriate directory and run `py ./main.py`. This will open up the UI which you are free to interact with and input any compatible historical dataset, however, I suggest following the steps shown in the below visual demonstration to prevent bugs:

https://user-images.githubusercontent.com/43887682/164047953-48d16dd3-5e12-4c83-bbd3-3e5ba85d60df.mp4

There are a few limitations of this solution:
- Live sentiment & fundamental analysis is not supported. This is due to the time taken to scrape a large number of webpages being too long to feasibly enforce live. Moreover, an AUS/USD dataset has been provided with pre-scraped news headlines and social media posts.
- The dashboard may exhibit slow behaviour during training and evaluation of models so please wait when evaluation models for the visualisations and metrics to appear.
