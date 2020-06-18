import matplotlib.pyplot as plot
import numpy as np
import structs
import infix as ops

# TODO: Add None-safety
# Syntactic sugar to add / investigate:
# Walrus operator (simplified assignment of throwaway vars)
# Iterator comprehensions
# Decorators
# Overloaded operators for monads?
# Function annotations --> attributes / argument validation

# Monad: a way to bind functions with computation results

class Figure_Container:
    # def __init__(self, title, title_attrs=default_title_attrs, dims=(1, 1)):
    #     self.__fig, axes, self.__dims = plot.subplots(nrows=dims[0], ncols=dims[1]), dims
    #     self.__fig.suptitle(title, fontweight=title_attrs.weight, fontsize=title_attrs.size,
    #         ha=title_attrs.ha, va=title_attrs.va)
    #     self.__subplot_it = (item for row in axes for item in row)
    #     self.__curr_subplot
    #     self.increment_subplot()
    def __init__(self, title, title_attrs=None, num_layers=1, add_by_row=True):
        self.__subplots, self.__title, self.__nrows, self.__byrow
            = [], title, num_layers, add_by_row
        self.__title_attrs = title_attrs <<ops.otherwise>> TextAttributes.subtitle()


    def add_scatterplots(self, *data, increment_subplots=False):

        return self
    def add_functions(self, *fns, increment_subplots=False):
        return self
    def add_new_subplot(self, title, title_attrs=None, is_xtwin=False, is_ytwin=False):
        self.__subplots.append({
            scatters: [],
            fns: [],
            annotations: [],
            comments: [],
            title: title,
            title_attrs: title_attrs,
            is_xtwin: is_xtwin,
            is_ytwin: is_ytwin
        })
        return self
    def show(self, tight_layout=True):
        # grid.legend(loc=legend_pos)
        if tight_layout:
            figure.tight_layout()
        plot.show()
        pass
    def __del__(self):





def initialize_figure(title, xlabel, ylabel):
    fig, grid = plot.subplots()
    plot.title(title, fontweight='bold', fontsize=14)
    grid.set_xlabel(xlabel)
    grid.set_ylabel(ylabel)
    plot.figtext(0.5, 0.01, caption, wrap=True, ha='center', va='top', fontsize=12)
    return fig, grid

def show_figure(figure, grid, legend_pos):
    grid.legend(loc=legend_pos)
    figure.tight_layout()
    plot.show()

def plot_points_with_errorbars(grid, data, marker, color, ebar_color, label, linestyle):
    grid.errorbar(x=data.x, y=data.y, xerr=None, yerr=data.ebars, ecolor=ebar_color,
        color=color, label=label, marker=marker, linestyle=linestyle)
    pass

def plot_function(grid, fn, color, label, linestyle, num_points=100):
    min, max = plot.xlim()
    fn_x = np.linspace(min, max, num_points)
    grid.plot(fn_x, fn(fn_x), color=color, label=label, linestyle=linestyle)
    pass

def annotate_points(grid, points, marker, color):
    for x, y, label in zip(points.x, points.y, points.labels):
        plot.annotate(label, (x, y), textcoords='offset points', xytext=(0,10), ha='center')
    pass

def annotate_point(grid, point, marker, color):
    plot.plot(point.x, point.y, color=color, marker=marker)
    plot.annotate(point.labels, (point.x, point.y), textcoords='offset points', xytext=(0,10), ha='center')
    pass

def initialize_new_figure(title, num_subplot_rows=1, num_subplot_cols=1):
    pass

def add_subplot_to_figure(fig):
    pass

def add_comments_to_subplot(sp):
    pass

def add_twin_to_subplot(sp):
    pass

def create_subplot(figure):
    pass

def show_all_figures(tight_layout=True):
    pass
