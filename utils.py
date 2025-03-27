#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:25:23 2025

@author: chiaramarzi
"""
##############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, average_precision_score, roc_auc_score, roc_curve
from sklearn.utils import resample
import xgboost as xgb

##############################################################################

# Function to print training and test metrics with 95% bootstrap intervals
def metrics_display(metrics_train, metrics_test, metric):
    """
    Print the average and 95% bootstrap intervals for a given metric.

    Parameters:
    -----------
    metrics_train : array-like of shape (M,)
        Metric values (e.g., accuracy) computed on the training set across M bootstrap iterations.

    metrics_test : array-like of shape (M,)
        Metric values computed on the test set across M bootstrap iterations.

    metric : str
        Name of the metric (e.g., 'Accuracy', 'Balanced accuracy') to be displayed in the output.
    """
    ave_score_train = np.mean(metrics_train)
    perc25_score_train = np.percentile(metrics_train, 2.5)
    perc975_score_train = np.percentile(metrics_train, 97.5)
    print("Train mean " + metric, ave_score_train, "BI 95 % [", perc25_score_train, ",", perc975_score_train, "]")
    
    ave_score_test = np.mean(metrics_test)
    perc25_score_test = np.percentile(metrics_test, 2.5)
    perc975_score_test = np.percentile(metrics_test, 97.5)
    print("Test mean " + metric, ave_score_test, "BI 95 % [", perc25_score_test, ",", perc975_score_test, "]")

##############################################################################

# Main analysis function for model training, evaluation, and SHAP analysis
def analysis(N, M, data, results_path, outfilename):
    """
    Perform classification using XGBoost with bootstrap validation and compute SHAP values.

    Parameters:
    -----------
    N : int
        Number of total samples (rows) in the dataset. Also the number of bootstrap samples drawn at each iteration.

    M : int
        Number of bootstrap iterations to perform (e.g., 1000).

    data : pandas.DataFrame
        The input dataset containing:
        - 'Patient' column (unique identifier)
        - 'Group' column (class label, e.g., IPF, NSIP, or COVID)
        - Radiomic features starting from 'shape_Elongation' onward

    outfilename : str
        Base name used for saving all output files (e.g., confusion matrices, metrics CSV, SHAP plots).
    """

    # Initialize the XGBoost classifier
    classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='mlogloss', use_label_encoder=False)

    # Initialize arrays to store performance metrics across M iterations
    accuracy_train_scores = np.zeros(M)
    accuracy_test_scores = np.zeros(M)
    bal_accuracy_train_scores = np.zeros(M)
    bal_accuracy_test_scores = np.zeros(M)
    average_precision_train_scores = np.zeros(M)
    average_precision_test_scores = np.zeros(M)
    roc_auc_train_scores = np.zeros(M)
    roc_auc_test_scores = np.zeros(M)
    C = np.zeros((2, 2, M))  # Confusion matrices for each iteration

    # Prepare data structures for mean ROC curve
    mean_fpr = np.linspace(0, 1, 1000)
    tprs = []
    df_ROC_coord = pd.DataFrame(data=mean_fpr, columns=['x'])

    # Begin bootstrap loop
    for m in range(M):
        # Stratified bootstrap resampling for training set
        train = resample(data, n_samples=N, replace=True, stratify=data['Group'], random_state=m)
        train_codes = train['Patient'].unique().tolist()
        test_codes = data.loc[~data['Patient'].isin(train_codes), 'Patient'].tolist()
        test = data.loc[data['Patient'].isin(test_codes), :]

        # Extract features and labels
        X_train = train.iloc[:, train.columns.tolist().index('feature_1'):]
        y_train = train['Group']
        X_test = test.iloc[:, test.columns.tolist().index('feature_1'):]
        y_test = test['Group']

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Predict probabilities and labels
        y_train_pred_proba = classifier.predict_proba(X_train)[:, 1]
        y_test_pred_proba = classifier.predict_proba(X_test)[:, 1]
        y_train_pred = classifier.predict(X_train)
        y_test_pred = classifier.predict(X_test)

        # Store performance metrics
        accuracy_train_scores[m] = accuracy_score(y_train, y_train_pred)
        accuracy_test_scores[m] = accuracy_score(y_test, y_test_pred)
        bal_accuracy_train_scores[m] = balanced_accuracy_score(y_train, y_train_pred)
        bal_accuracy_test_scores[m] = balanced_accuracy_score(y_test, y_test_pred)
        average_precision_train_scores[m] = average_precision_score(y_train, y_train_pred_proba)
        average_precision_test_scores[m] = average_precision_score(y_test, y_test_pred_proba)
        roc_auc_train_scores[m] = roc_auc_score(y_train, y_train_pred_proba)
        roc_auc_test_scores[m] = roc_auc_score(y_test, y_test_pred_proba)
        C[:, :, m] = confusion_matrix(y_test, y_test_pred, normalize='true')

        # Compute and store interpolated ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        df_ROC_coord['y_' + str(m)] = interp_tpr

    # Save ROC curve data
    df_ROC_coord.to_csv(results_path + "/" + outfilename + "_ROC_coords_test.csv", index=False)

    # Train classifier on full dataset for SHAP
    X = data.iloc[:, train.columns.tolist().index('feature_1'):]
    y = data['Group']
    classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='mlogloss', use_label_encoder=False)
    classifier.fit(X, y)

    # Compute and plot SHAP feature importance
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(
        shap_values, X.values,
        plot_type="bar",
        class_names=['COVID', 'NSIP'],
        feature_names=X.columns.tolist(),
        max_display=10,
        show=False
    )
    plt.xlabel("Mean(|SHAP value|)")
    plt.savefig(results_path + '/' + outfilename + '-SHAP_summary.png', dpi=600, format='png')

    # Save metrics to CSV
    pd.DataFrame(list(zip(
        accuracy_train_scores, bal_accuracy_train_scores,
        average_precision_train_scores, roc_auc_train_scores)),
        columns=['Accuracy', 'Balanced accuracy', 'Average precision', 'AUROC']
    ).to_csv(results_path + "/" + outfilename + "-Train_scores.csv", index=False)

    pd.DataFrame(list(zip(
        accuracy_test_scores, bal_accuracy_test_scores,
        average_precision_test_scores, roc_auc_test_scores)),
        columns=['Accuracy', 'Balanced accuracy', 'Average precision', 'AUROC']
    ).to_csv(results_path + "/" + outfilename + "-Test_scores.csv", index=False)

    # Save confusion matrices
    np.save(results_path + '/' + outfilename + '-Confusion_matrices.npy', C)

    # Print summary metrics
    metrics_display(accuracy_train_scores, accuracy_test_scores, "Accuracy")
    metrics_display(bal_accuracy_train_scores, bal_accuracy_test_scores, "Balanced accuracy")
    metrics_display(average_precision_train_scores, average_precision_test_scores, "Average precision")
    metrics_display(roc_auc_train_scores, roc_auc_test_scores, "AUROC")

##############################################################################

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          vmin=None,
                          vmax=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cf using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    percent:       If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'.

    title:         Title for the heatmap.

    vmin, vmax:    Minimum and maximum values to scale the colormap.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        accuracy = np.trace(cf) / float(np.sum(cf))

        if len(cf) == 2:  # Binary confusion matrix
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories,
                vmin=vmin, vmax=vmax)  # use vmin e vmax

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)