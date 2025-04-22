import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as PathMat
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D


def radial_chart(data, spoke_labels, labels, colors, markers, title, save=None):
    N = len(spoke_labels)
    theta = radar_factory(N, frame='polygon')

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
    # fig.patch.set_facecolor('#203864')  # Sets entire figure background
    # ax.set_facecolor('#203864')
    # ax.yaxis.grid(True, color='white', linewidth=0.5)  # Changes radial grid lines to red
    # ax.xaxis.grid(True, color='black', linewidth=0.5)  # Changes radial grid lines to red
    ax.xaxis.grid(False)
    # ax.grid(False)
    # ax.yaxis.grid(False)    # Hide radial grid lines
    # ax.grid(False)          # Hide circular grid lines
    # ax.set_yticklabels([])
    # Plot the four cases from the example data on separate Axes
    # for ax, (title, case_data) in zip(axs.flat, data):
    # ax.set_rgrids([0.3, 0.5, 0.9])
    case_data = data
    # ax.set_title(title, weight='bold', size=50, position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center', color="white")
    for d, color, marker in zip(case_data, colors, markers):
        # ax.plot(theta, d, color='black', linewidth=2)
        # if color == '#007D8C':
        #     alpha = 0.6
        # elif color == '#D26432':
        #     alpha = 0.6
        # elif color == '#C89B37':
        #    alpha = 0.6
        # ax.fill(theta, d, facecolor=color, alpha=alpha, label='_nolegend_')
        ax.plot(theta, d, color=color, label='_nolegend_', linewidth=3)
    ax.plot(theta, [0.6] * N, color='r', linestyle='-.', label='_nolegend_')

    # ax.set_varlabels(spoke_labels,  fontsize=20, color='white')
    ax.set_varlabels(spoke_labels, fontsize=20)
    # ax.tick_params(axis='y', colors='lightgray')  # Change grid label colors
    # ax.spines['polar'].set_color('white')
    # ax.tick_params(axis='y', labelsize=15, labelcolor='white')  # Adjust the number to your preference
    ax.tick_params(axis='y', labelsize=15)  # Adjust the number to your preference

    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # Define where the ticks should appear
    # ax.set_yticklabels(['Low', 'Medium', 'High'], fontsize=12, color='white')
    # add legend relative to top-left plot

    # Custom legend handles: create square patches
    legend_handles = [
        Line2D([0], [0], color=color, marker='p', markersize=30, linestyle='None', markeredgewidth=3, fillstyle='none')
        for color in colors]

    legend = ax.legend(legend_handles, labels, loc=(0.9, .95),
                       labelspacing=0.1, fontsize=30)

    fig.text(0.5, 0.965, '',
             horizontalalignment='center', color='black', weight='bold',
             size='large')

    plt.show()
    if save is not None:
        plt.savefig(save)


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return PathMat(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize='small', color='black'):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize, color=color)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=PathMat.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta