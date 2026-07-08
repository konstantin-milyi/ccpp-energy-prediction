import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error
)


def plot_feature_diagnostics(df, features, cols=3):
    """
    Generates a set of plots for feature distribution analysis.

    For each variable in the `features` list, a linked pair of plots is created: 
    top — a histogram with skewness in the title; bottom — a boxplot with a synchronized X-axis.
    Empty subplots in the grid are automatically removed.

    Args:
        df (pandas.DataFrame): The input dataframe.
        features (list): A list of column names (strings).
        cols (int, optional): Number of columns in the final plot grid. Defaults to 3.

    Returns:
        None: The function returns nothing; it displays the plots on the screen.
    """
    
    # Number of rows
    n_rows = math.ceil(len(features) / cols)
    
    # Create the grid
    fig, axes = plt.subplots(
        n_rows * 2, # Two rows per feature (histogram + boxplot)
        cols, 
        figsize=(15, 5 * n_rows), # Height scales proportionally with the number of rows
        gridspec_kw={'height_ratios': [4, 1] * n_rows} # Alternate heights: 4 parts for histogram, 1 part for boxplot
    )
    
    # Reshape the axes array to the desired form
    axes = axes.reshape(-1, cols)
    
    # Loop through the slots in the grid
    for i in range(n_rows * cols):
        
        # Calculate axis indices for the specific feature 
        ax_hist = axes[(i // cols) * 2, i % cols]      # Axis for the histogram (top)
        ax_box  = axes[(i // cols) * 2 + 1, i % cols]  # Axis for the boxplot (1 row below)
        
        # While the list of variables is not exhausted
        if i < len(features):
            var = features[i] # Get the current column name
            
            # Histogram
            sns.histplot(data=df, x=var, kde=True, ax=ax_hist)
            ax_hist.set_title(f"{var} (skewness: {df[var].skew():.2f})", fontweight='bold')
            
            # Remove X-axis label
            ax_hist.set(xlabel='')
            # Hide X-axis tick labels for the histogram
            ax_hist.tick_params(labelbottom=False) 
            
            # Boxplot
            sns.boxplot(data=df, x=var, ax=ax_box, fliersize=3)
            # Set X-axis label for the bottom plot
            ax_box.set(xlabel=var)
            
            # Synchronize X-axis scale between histogram and boxplot
            ax_hist.sharex(ax_box)
            
        else:
            # If features run out but empty cells remain, remove these axes
            ax_hist.remove()
            ax_box.remove()

    plt.tight_layout()
    plt.show()


def find_boundaries(df, variable, distance):
    """
    Calculates the lower and upper boundaries for identifying outliers 
    using the Interquartile Range (IQR) method.

    Parameters:
        df (pandas.DataFrame): The input dataframe.
        variable (str): The name of the variable (column) for which boundaries are calculated.
        distance (float): The IQR multiplier that determines the "strictness" of the boundaries 
                          (typically 1.5 for standard outliers and 3.0 for extreme outliers).

    Returns:
        tuple: A tuple containing two numeric values (lower_boundary, upper_boundary).
    """
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)

    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)

    return lower_boundary, upper_boundary


def _get_metrics(y_true, y_pred):
    """
    Calculates the main quality metrics for a regression task.

    Parameters:
    ----------
    y_true (array-like): Actual (true) values of the target variable.
    y_pred (array-like): Predicted values of the target variable.
    ----------
    Returns: A dictionary containing the calculated metrics.
    """
    return {
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100
    }


def model_metrics_barplot(model_name, model, X_train, y_train, X_test, y_test, ax=None, barlabel_siaze=10):
    """
    Calculates model metrics and plots a bar chart for training and test sets.

    Parameters:
    ----------
    model_name (str): Name of the ML model (chart title).
    model (estimator): Trained ML model.
    ax (matplotlib.axes.Axes), default None: Axes object (grid subplot). 
        If None, the function creates a new independent canvas.
    barlabel_siaze (int), default 10: Font size for metric values (above bars).
    ----------
    Returns: The function returns nothing, but displays the plot on the screen.
    """
    # Bind metrics to a DataFrame
    df_metrics = pd.DataFrame({
        "Train": _get_metrics(y_train, model.predict(X_train)),
        "Test": _get_metrics(y_test, model.predict(X_test))
    })

    color = sns.color_palette("deep").as_hex()
    
    # Check if ax is provided 
    is_standalone = (ax is None)
    
    # Plot parameters
    plot_kws = dict(color=[color[0], color[1]], edgecolor='black', rot=0, title=model_name)
    
    if is_standalone:
        plot_kws['figsize'] = (9, 5) # Size of the standalone plot
        
    ax = df_metrics.plot.bar(ax=ax, **plot_kws)
    
    # Styling
    ax.title.set(fontweight='bold', fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle='--', alpha=0.6)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.05)
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', padding=3, fontsize=barlabel_siaze)

    # Call show() only if the plot is standalone 
    if is_standalone:
        plt.tight_layout()
        plt.show()


def show_final_history(history):
    """
    Visualizes the neural network training process over epochs.

    Builds a grid of three plots (Loss, MAE, and R2), displaying curves 
    for the training (Train) and validation (Validation) sets. 
    Helps to visually assess the model's convergence speed and the presence of overfitting.

    Parameters:
    ----------
    history : keras.callbacks.History
        Training history object returned by the `model.fit()` function 
        in Keras/TensorFlow. Inside it has a dictionary with the following keys: 
        'loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error', 
        'r2_score', 'val_r2_score'.
    -----------
    Returns: The function returns nothing, but displays the finished plot on the screen.
    """
    # Map training metric, validation metric, and display title
    metrics = [
        ('loss', 'val_loss', 'Loss'),
        ('mean_absolute_error', 'val_mean_absolute_error', 'MAE'),
        ('r2_score', 'val_r2_score', 'R2 Score')
    ]
    
    # 1x3 Grid 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    # Build plots 
    for ax, (train_col, val_col, title) in zip(axes, metrics):

        ax.plot(history.history[train_col], label='Train', color='#4C72B0', lw=2)
        ax.plot(history.history[val_col], label='Validation', color='#DD8452', lw=2)
        
        ax.set(title=title, xlabel='Epoch')
        ax.legend()
        ax.grid(linestyle='--', alpha=0.6)
        
    plt.show()


def plot_pred_vs_true(y_true, y_pred, title="Predicted vs True", ax=None):
    """
    Visualizes the quality of the model's predictions using a scatter plot.

    Plots actual values on the X-axis and predicted values on the Y-axis.
    Adds a diagonal red line for ideal predictions (y = x).

    Parameters:
    ----------
    ax (matplotlib.axes.Axes), default None:
        Axes object (grid subplot). If None, the function creates a new 6x6 canvas.
    -----------
    Returns: The function returns nothing, displays the plot on the screen.
    """
    # Check if ax is provided
    is_standalone = (ax is None)
    if is_standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Calculate axis limits
    lims = [np.min([y_true, y_pred]), np.max([y_true, y_pred])]
    
    # Plot points and diagonal 
    ax.scatter(y_true, y_pred, s=12, alpha=0.5)
    ax.plot(lims, lims, lw=2, color='red')  
    
    # Configure axes and titles 
    ax.set(
        xlim=lims, ylim=lims,
        xlabel="True", ylabel="Predicted",
        title=f"{title} (R2={r2_score(y_true, y_pred):.3f})"
    )
    
    # Styling
    ax.title.set(fontweight='bold')
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3, linestyle='--') 
    
    # Call show() only if the plot is standalone 
    if is_standalone:
        plt.tight_layout()
        plt.show()


def plot_residuals_hist(y_true, y_pred, title="Residuals Histogram", bins=30, ax=None):
    """
    Visualizes the distribution of model residual errors using a histogram.

    Parameters:
    ----------
    y_true : Actual values of the target variable.
    y_pred : Values predicted by the model (expected to match the size of y_true).
    title : Chart title.
    bins (int): Number of bins (columns) to divide the error range into.
    ax (matplotlib.axes.Axes), default None: Axes object (grid subplot). 
    If None, the function creates a new canvas.
    -----------
    Returns: The function returns nothing, displays the plot on the screen.
    """
    resid = np.asarray(y_true) - np.asarray(y_pred)

    # Check if ax is provided 
    is_standalone = (ax is None)
    if is_standalone:
        fig, ax = plt.subplots(figsize=(6.5, 4))

    # Build histogram
    sns.histplot(resid, bins=bins, kde=True, ax=ax)

    # Configure axes and titles 
    ax.set_xlabel("Residual (y_true - y_pred)")
    ax.set_title(title, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--') 
    
    # Show plot only if it's standalone
    if is_standalone:
        plt.tight_layout()
        plt.show()