from netCDF4 import Dataset
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, FastICA
import pandas as pd
import geopandas as gpd
import cartopy.io.shapereader as shpreader
from tqdm.notebook import tqdm
from sklearn.base import clone


def plot_maps(temperature, lon, lat):
    # Plots two subplots side by side
    fig, axes = plt.subplots(1, len(temperature), figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    for kk in range(len(temperature)):
        axes[kk].set_global()
        axes[kk].coastlines()
        axes[kk].contourf(lon, lat, temperature[kk],transform=ccrs.PlateCarree(), cmap='coolwarm') 
        
    return fig, axes


def load_simulation(set='R1'):
    '''
    Loads one of the simulation data sets
    '''
    R1 =  Dataset(f'data/T2m_{set}_ym_1stMill.nc', 'r')
    temperature = R1.variables['T2m'][:]
    lat = R1.variables['lat'][:]
    lon = R1.variables['lon'][:]

    return temperature, lat, lon


def sensor_fields(nrows, ncols, divisor_row=6, divisor_col=6):
    '''
    Creates a grid of size nrows, ncols and sperates it into fields
    by dividing nrows by divisior_row, etc.
    '''
    lan_division = np.int(ncols / divisor_col)
    lon_division = np.int(nrows / divisor_row)

    sensor_grid = np.zeros((nrows, ncols))
    c = 0
    for ii in range(divisor_col):
        for jj in range(divisor_row):
            sensor_grid[jj * lon_division : (jj+1) * lon_division, 
                        ii * lan_division : (ii+1) * lan_division] = c
            c += 1
    return sensor_grid


def fit_on_grid(X, y, clf, grid):
    '''
    Train multiple classifiers based on the grid information
    '''
    clf_list = []
    for gg in np.unique(grid):
        # Create a new classifier each round.
        clf_list.append(clone(clf))
        clf_list[-1].fit(X[:, grid==gg], y)
    
    return clf_list


def predict_on_grid(X, y, clf_list, grid, proba=True):
    '''
    Predicts using multiple classifier
    '''
    # Instatniate
    predictions = np.zeros((y.shape[0], np.unique(grid).shape[0]))
    
    # Loop over grid
    for clf, gg in zip(clf_list, np.unique(grid)):
        if proba:
            predictions[:, gg.astype(int)] = clf.predict_proba(X[:, grid==gg])[:, 1]
        else:
            predictions[:, gg.astype(int)] = clf.predict(X[:, grid==gg])
    return predictions


def score_on_grid(y, y_pred, metric):
    '''
    Calculates a score for each prediction of a grid classifier.
    '''
    scores = np.zeros(y_pred.shape[1])
    for ii in range(y_pred.shape[1]):
        scores[ii] = metric(y, y_pred[:, ii])
        
    return scores