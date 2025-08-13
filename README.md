# 🌲 Forest Cover Type Prediction using Machine Learning

In this project 4 different supervised machine learning models were used to predict forest cover types in 4 different forest belonging to the **Roosevelt National Forest** in Colorado, USA.

The dataset used in this project can be accessed both in [Kaggle](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset?resource=download) and University of California, Irvine's ML [Repository](https://archive.ics.uci.edu/dataset/31/covertype). The dataset comes from *Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables*, an article published in *Computers and Electronics in Agriculture*.

## Objectives:
This project tries to acomplish two things:
- Comparing different supervised machine learning algorithms for classification in order to study the performance of each model and check which model best suits a classification problem in a relatively complex dataset.
- Surpass the prediction capabilities of an Artificial Neural Network used in the article.

## Contents/Structure:
The structure of the repository is the following (only directories with useful files will be described, the rest are there for support purposes):
```
notebooks/
|---data_cleansing_&_EDA.ipynb  # A first glimpse to the dataset and the distribution and properties of some variables, including the target variable.
|---models.ipynb                # Notebook where the different models (*KNN, Decision Trees, Random Forest and AdaBoost* classifieres) used in this project are compared and their properties analyzed. In addition, an unsupervised learning algorithm is also used (*PCA analysis*) for dimensionality reduction, and the effects of neglecting some of the categorical features are analyzed.
data/
|---raw/
    |---covtype.csv              # Data stored in CSV format.
    |---covtype.info             # Information about the data. A shorter version of the information that can be found in the article.
app/
|---app.py                       # An app made using streamlit to provide a visual interface by which introduce the necessary data to predict the forest cover types of new plot of land.
```

## Results and conclusions:

1. *K-Nearest Neighbours* is not suited for highly dimensional datasets and offers poor results.
2. *Decision Tree classifiers* offer good metrics with low computational cost and show better results than the *Gaussian Discriminant Analysis* model.
3. The two ensemble models used, *Random Forest* and *Ada Boost*, show **better results than the *Artificial Neural Network* used by the authors**! This is quite impressive, and shows that under certain circunstances, *classical* machine learning algorithms can match or even surpass the results of Neural Networks!
4. Dimensionality reduction has not been able to improve the metrics in any way.

## Dependencies:

- Pandas
- Matplotlib
- Seaborn
- Scikit-learn