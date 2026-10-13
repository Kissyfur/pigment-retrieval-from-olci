import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_heat_map(data, col_name, title='', label='', log_scale_bar=True, save=None):
    # Create a figure with a map projection
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.Mercator())  # You can choose different projections like 'PlateCarree'

    # Add coastlines and other features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, zorder=10, edgecolor='black')
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)

    # Set the extent for the Mediterranean Sea
    ax.set_extent([-6., 36, 30, 46], crs=ccrs.PlateCarree())  # [west, east, south, north]

    # Plot the data as a scatter plot
    sc = ax.scatter(data['lon'], data['lat'], c=data[col_name], cmap='viridis', s=0.001, transform=ccrs.PlateCarree(),
                    norm=matplotlib.colors.LogNorm() if log_scale_bar else None)

    # Add a colorbar
    cbar = plt.colorbar(sc, orientation='vertical', pad=0.02, shrink=0.5)
    cbar.set_label(label)

    # Add title and labels
    plt.title(title, fontsize=16)

    # Show or save the plot
    plt.savefig(save, dpi=300) if save is not None else None
    # plt.show()
    plt.close()

def plot_lines(lines, title, save=None):
    # max_y = max([len(line) for line_name, line in lines.items()])
    for line_name, line in lines.items():
        # if len(line) == 1:
        #     line = line + [line[-1]] * (max_y - len(line))
        x = range(len(line))
        plt.plot(x, line, label=line_name)
    plt.title(title)
    plt.legend()
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_mu_sigma_metric(met, classes, title, save=None):
    for name, met_d in met.items():
        mu = np.array(met_d['mu'])
        si = np.array(met_d['sigma'])
        plt.errorbar(classes, mu, si, label=name, solid_capstyle='projecting', capsize=5,
                     linestyle='-' if 'LR' in name else ':')
    plt.title(f'Metric {title}')
    plt.grid(alpha=0.5, linestyle=':')
    plt.xticks(rotation=90)
    plt.legend()
    if save is not None:
        plt.savefig(save)
    plt.show()


def plot_stack(y, py, labels, title):
    x = np.arange(y.shape[0])
    # Create a larger figure for side-by-side charts
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    # Plot the model predictions on the left
    axs[0].stackplot(x, py.T, alpha=0.6, labels=labels)
    axs[0].legend(loc='upper right')
    axs[0].set_title('Model: ' + title)
    axs[0].set_xlabel('sample')
    axs[0].set_ylabel('Predicted values')

    # Plot the true values on the right
    axs[1].stackplot(x, y.T, alpha=0.6, labels=labels)
    axs[1].legend(loc='upper right')
    axs[1].set_title('True values')
    axs[1].set_xlabel('sample')
    axs[1].set_ylabel('True values')

    # Adjust spacing between the two subplots
    plt.subplots_adjust(wspace=0.3)

    # Show the figure
    plt.show()

