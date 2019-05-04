# coding=utf8
'''
ref: https://github.com/drazenz/heatmap/blob/master/Circle%20heatmap.ipynb
'''
import os
import urllib
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def download_data(url, data_dir):
    req = requests.get(url)
    if req.status_code == 404:
        print("No file exists.")
        return
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    filename = url.split('/')[-1]
    data_path = os.path.join(data_dir, filename)
    print(data_path)
    with open(data_path, 'wb') as f:
        f.write(req.content)
    print("download over.")

def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

def corr_mat_plot(data):
    corr = data.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(220,20, n=200),
        square=True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 45,
        horizontalalignment = 'right')
    plt.show()

def scatter_heatmap(x, y, size, color):
    fig, ax = plt.subplots()
    
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x2num = {p[1]:p[0] for p in enumerate(x_labels)}
    y2num = {p[1]:p[0] for p in enumerate(y_labels)}
    size_scale = 500

    n_colors = 256
    palette = sns.diverging_palette(220, 20, n=n_colors)
    color_min, color_max = [-1, 1]
    def val2color(val):
        # .0 <= val <= 1.0
        val = float((val - color_min) / (color_max - color_min))
        ind = int(val * (n_colors - 1))
        return palette[ind]

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x15 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the leftmost 14 columns of the grid for the main plot

    ax.scatter(
        x = x.map(x2num),
        y = y.map(y2num),
        s = size * size_scale,
        c = color.apply(val2color),
        marker = 's')
    ax.set_xticks([x2num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation = 45, horizontalalignment = 'right')
    ax.set_yticks([y2num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    # move squares to center
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    # fix the left and bottom side
    ax.set_xlim([-0.5, max([v for v in x2num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y2num.values()]) + 0.5])

    # add color legend bar
    ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

    col_x = [0]*len(palette) # Fixed x coordinate for the bars
    y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

    bar_height = y[1] - y[0]
    ax.barh(
        y=y,
        width=[5]*len(palette), # Make bars 5 units wide
        left=col_x, # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False) # Hide grid
    ax.set_facecolor('white') # Make background white
    ax.set_xticks([]) # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(y), max(y), 3)) # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right() # Show vertical ticks on the right
    
    plt.show()

def scatter_heatmap_test(data):
    columns = ['bore', 'stroke', 'compression-ratio', 'horsepower', 'city-mpg', 'price']
    corr = data[columns].corr()
    corr = pd.melt(corr.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    scatter_heatmap(
        x=corr['x'],
        y=corr['y'],
        size=corr['value'].abs(),
        color=corr['value'])

if __name__ == "__main__":
    data_dir = '../data'
    url = 'https://raw.githubusercontent.com/drazenz/heatmap/master/autos.clean.csv'
    filename = url.split('/')[-1]
    data_path = os.path.join(data_dir, filename)
    # download_data(url, data_dir)
    data = read_data(data_path)
    #corr = data.corr()
    #sns.heatmap(corr)
    #plt.show()
    corr_mat_plot(data)
    # scatter_heatmap_test(data)



    
