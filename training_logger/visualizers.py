import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Iterable
from training_logger import TrainingLogger
from PIL import Image

class LogVisualizer:
    """
    Class to display training progress of a single run
    """
    
    def __init__(self, path, prefix=''):
        """
        Create a new :class:`LogVisualizer`
        
        :param path: The path of the directory containing the run log and metadata
        :param prefix: Prefix to add to labels when plotting. This is useful when plotting data from more than one source on the same axes object
        """
        self.logger = TrainingLogger(path, True)
        self.prefix = prefix
    
    def update_data(self):
        """
        Reread the data from the training directory.
        
        Note that you must call all plot functions again to show the updated data, this method
        only updates the internal state of the visualizer.
        """
        try:
            self.logger = TrainingLogger(self.logger.basename, overwrite=True)
        except Exception as e:
            print("Could not update...")
            print(str(e))
        
    def smooth_data(self, data, window_size, sigma):
        window = np.arange(-(window_size // 2), window_size // 2 + 1)
        window = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- 1 / 2 * (window / sigma) ** 2)
        window /= np.sum(window)

        to_conv = np.zeros((data.shape[0] + window_size - 1))
        to_conv[window_size // 2:-(window_size // 2)] = data
        to_conv[:window_size // 2] = data[0]
        to_conv[-window_size // 2:] = data[-1]

        return np.convolve(to_conv, window, mode='valid')
    
    def show_graph(self, name, axes=None, ylim=None, xlim=None, smooth_window=0, smooth_sigma=3, legend=True, **kwargs):
        """
        Base method to plot a scalar value.
        
        :param name: The name of the column to plot data from. This must be a scalar column.
        :param axes: The axes object to plot on. If None, a new figure is created and it's axes used.
        :param ylim: Y-limit of the plot
        :param xlim: X-limit of the plot
        :param smooth_window: Window to average for smoothing the plot. If 0, no smoothing is performed, otherwise
                    the data is smoothed as :math:`\\hat{x}[i] = \\frac{\\sum_{j = -\\left\\lfloor s/2 \\right\\rfloor}^{\\left\\lfloor s/2 \\right\\rfloor} x[i + j]}{s}`, where :math:`s`
                    is the smoothing window.
        :param kwargs: Keyword-arguments passed to :meth:`Axes.plot()<matplotlib.axes.Axes.plot>`. Should not contain label.
        :returns: :class:`~matplotlib.axes.Axes` object, the one used to draw the plot.
        :raises: :exc:`AssertionError` if name is not a scalar column
        """
        assert name in self.logger.data.columns
        assert self.logger.metadata[name] == 'scalar'
        plt_data = self.logger.data[name].dropna()

        x = plt_data.index.values
        y = plt_data.values

        if smooth_window > 0:
            y = self.smooth_data(y, smooth_window, smooth_sigma)

        if axes is None:
            axes = plt.figure().gca()
        axes.plot(x, y, label=self.prefix + name, **kwargs)
        if legend:
            axes.legend()
        if ylim is not None:
            axes.set_ylim(ylim)
        if xlim is not None and isinstance(xlim, Iterable):
            axes.set_xlim(xlim)
        elif xlim is not None:
            axes.set_xlim((xlim, self.logger.data.index.max()))
        return axes
    
    def show_img(self, name, iteration, axes=None):
        """
        Base method to show an image from the training log.
        
        :param name: The name of the column to plot the image from. This must be an image column.
        :param iteration: The iteration to show the image from.
        :param axes: The axes object to draw on. If None, a new figure is created and it's axes used.
        :returns: :class:`~matplotlib.axes.Axes` object, the one used to draw the image.
        :raises: :exc:`AssertionError` if name is not an image column
        :raises: :exc:`FileNotFoundError` if the image has been moved or removed from the training directory
        """
        assert name in self.logger.data.columns
        assert self.logger.metadata[name] == 'img'

        if axes is None:
            fig = plt.figure(figsize=(30,6))
            
            axes = plt.axes()
        
        path = self.logger.data.loc[iteration, name]
        assert path is not None
        i = Image.open(path)
        a = np.asarray(i)

        axes.imshow(a)
        axes.set_title(self.prefix + name)
        i.close()
        return axes
    
    def show_matching_scalars(self, expr, axes=None, **kwargs):
        """
        Show plots of all scalar columns that matches an expression.
        
        This function can take either a single expression or multiple.
        Unless an axes is specified, the method plots using the following directions:
        If multiple expressions are given, the columns matching each expression
        are plotted in the same axes, while columns from different expressions
        are plotted in different axes. If only one expression is given, all columns
        are plotted in the same axes.
        
        :param expr: String or list of strings, the expression(s) to match agains. Should follow standard Python :mod:`re` syntax.
        :param axes: The axes to plot. If not None, all values are plotted in this one. Otherwise, see the description above.
        :param kwargs: Keyword-arguments passed to :func: :meth:`~LogVisualizer.show_scalars`.
        :returns: :class:`~matplotlib.axes.Axes` object, the one used to draw the plot.
        """
        if not isinstance(expr, Iterable):
            expr = [expr]
        names = []
        for ex in expr:
            for col_name in self.get_cols():
                if re.fullmatch(ex, col_name):
                    names.append(col_name)
        axes = self.show_scalars(names, False, axes, **kwargs)
        return axes
    
    def show_scalars(self, names, subplots=True, axes=None, **kwargs):
        """
        Method to show multiple scalars.
        
        This method simply calls :meth:`~LogVisualizer.show_graph` for all names,
        as well as some handling of the axes objects.
        
        :param names: Iterable of column names. Each must be a scalar column.
        :param subplots: If true, each column is plotted on its own axis. Otherwise, they are plotted on the same.
        :param axes: The :class:`~matplotlib.axes.Axes` object used to plot the (first) data series
        :param kwargs: Keyword-arguments passed to :meth:`~LogVisualizer.show_graph`
        :returns: :class:`~matplotlib.axes.Axes` object, the one used to draw the plot.
        """
        for name in names:
            axes = self.show_graph(name, axes, legend=subplots, **kwargs)
            if subplots:
                axes = None
        if not subplots:
            axes.legend()
        return axes
    
    def show_all_scalars(self, subplots=True, axes=None, **kwargs):
        """
        Method for showing all available scalars.
        
        :param subplots: If True, each column is plotted on its own axis. Otherwise, they are plotted on the same.
        :param axes: The :class:`~matplotlib.axes.Axes` object used to plot the (first) data series
        :param kwargs: Keyword-arguments passed to :meth:`~LogVisualizer.show_graph()`
        :returns: :class:`~matplotlib.axes.Axes` object, the one used to draw the plot.
        """
        for k, v in self.logger.metadata.items():
            if v == 'scalar':
                axes = self.show_graph(k, axes, **kwargs)
                if subplots:
                    axes = None
    
    def get_cols(self):
        """
        :returns: A list containing all column names
        """
        return list(self.logger.data.columns)
    
    def get_non_null_index(self, col):
        """
        Get all indices where the value for the column :code:`col` is not null/na.
        
        :returns: A list of the non-null indices for column col
        """
        return list(self.logger.data[col].dropna().index)

class MultiLogVisualizer:
    """
    Class for comparing the training progress of multiple runs.
    
    Wraps an arbitrary number of :class:`LogVisualizer`
    """
    
    def __init__(self, *paths):
        """
        Create a new MultiLogVisualizer with the logs specified by paths
        
        :param paths: One or more paths to log dirs
        """
        self.visualizers = [LogVisualizer(path, f"{path.rpartition('/')[-1]}/") for path in paths]
        
    def update_data(self):
        """
        Update the data for all internal visualizers.
        """
        for viz in self.visualizers:
            viz.update_data()
    
    def show_graph(self, name, axes=None, ylim=None, xlim=None, **kwargs):
        """
        You should never try to show the graph directly from the MultiLogVisualizer.
        Instead, use one of the higher-level methods which calls show_graph on
        the internal visualizers.
        """
        raise NotImplementedError()
    
    def show_img(self, name, iteration, axes=None):
        """
        Show an image-grid of the images from all visualizers.
        
        The image grid has dimensions height = floor(sqrt(n)),
        and width = ceil(n / height), where n is the number of
        images to show.
        
        :param name: Name of column to show. This should be an image column. Passed to the show_img method of each internal visualizer.
        :param iteration: The iteration to show images from. Passed to the show_img method of each internal visualizer.
        :param axes: The axes to show images on. Passed to the show_img method of each internal visualizer.
        """
        n_img = len(self.visualizers)
        height = int(np.floor(np.sqrt(n_img)))
        width = int(np.ceil(n_img / height))
        fig, ax = plt.subplots(height, width, squeeze=False)
        for i, viz in enumerate(self.visualizers):
            axes = ax[i // width, i % width]
            viz.show_img(name, iteration, axes)
        return axes
    
    def show_matching_scalars(self, expr, axes=None, **kwargs):
        """
        Show plots of all scalar columns that matches an expression.
        
        This function can take either a single expression or multiple.
        Dataseries from the same expression from all visualizers are plotted in the same axes object.
        Dataseries from different expressions are plotted in different axes.
        
        :param expr: String or list of strings, the expression(s) to match agains. Should follow standard Python :mod:`re` syntax.
        :param axes: The :class:`~matplotlib.axes.Axes` to plot the first expression in.
        :param kwargs: Keyword-arguments passed to the show_graph method of internal visualizers.
        :returns: :class:`~matplotlib.axes.Axes` object, the one used to draw the final dataseries.
        """
        if not isinstance(expr, Iterable):
            expr = [expr]
        for ex in expr:
            names = set()
            for col_name in self.get_cols():
                if re.fullmatch(ex, col_name):
                    names.add(col_name)
            for viz in self.visualizers:
                for name in names:
                    try:
                        axes = viz.show_graph(name, axes, **kwargs)
                    except AssertionError:
                        pass
            axes = None
        return plt.gca()
        
    
    def show_scalars(self, names, subplots=True, axes=None, **kwargs):
        """
        Method to show multiple scalars.
        
        This method simply calls :meth:`LogVisualizer.show_graph` for all combination of names and internal visualizers,
        as well as some handling of the axes objects.
        
        :param names: Iterable of column names. Each must be a scalar column.
        :param subplots: If true, each column is plotted on its own axis. The same column from different visualizers are plotted together. 
                        If false, all columns are plotted together.
        :param axes: The :class:`~matplotlib.axes.Axes` object used to plot the (first) data series
        :param kwargs: Keyword-arguments passed to :meth:`LogVisualizer.show_graph`
        :returns: :class:`~matplotlib.axes.Axes` object, the one used to draw the plot.
        """
        for name in names:
            for viz in self.visualizers:
                try:
                    axes = viz.show_graph(name, axes, **kwargs)
                except AssertionError:
                    pass
            if subplots:
                axes = None
    
    def show_all_scalars(self, subplots=True, axes=None, **kwargs):
        """
        Method for showing all available scalars from all visualizers.
        
        :param subplots: If true, each column is plotted on its own axis. The same column from different visualizers are plotted together. 
                        If false, all columns are plotted together.
        :param axes: The :class:`~matplotlib.axes.Axes` object used to plot the (first) data series
        :param kwargs: Keyword-arguments passed to :meth:`LogVisualizer.show_graph()`
        :returns: :class:`~matplotlib.axes.Axes` object, the one used to draw the plot.
        """
        md = self.visualizers[0].metadata
        for viz in self.visualizers[1:]:
            md.update(viz.metadata)
        for k, v in md.items():
            if v == 'scalar':
                for viz in self.visualizers:
                    try:
                        axes = viz.show_graph(k, axes, **kwargs)
                    except AssertionError:
                        pass
                if subplots:
                    axes = None
        return plt.gca()
    
    def show_category_scalars(self, category, *args, single_axis = True, **kwargs):
        orig_axes = kwargs.pop('axes', None)
        return self.show_matching_scalars(f"{category}/", orig_axes, **kwargs)
    
    def get_cols(self):
        """
        :returns: A list containing all column names
        """
        return list(set([c for viz in self.visualizers for c in viz.get_cols()]))
    
    def get_non_null_index(self, col):
        """
        Get all indices where the value for the column :code:`col` is not null/na.
        
        :returns: A list of list of the non-null indices for column col. Each internal list corresponds to the visualizer at the same position.
        """
        return [viz.get_non_null_index(col) for viz in self.visualizers]
