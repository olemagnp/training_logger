import pandas as pd
import numpy as np
import os, sys, shutil
import json
from PIL import Image
from matplotlib import pyplot as plt
from collections import Iterable

class TrainingLogger:
    DATATYPES = set({"scalar", "img"})
    def __init__(self, basename, overwrite=False):
        self.basename = basename
        try:
            self.data = pd.read_csv(os.path.join(basename, "data.csv"), index_col=0)
            with open(os.path.join(basename, "data.meta"), "r") as f:
                json_str = f.read().replace("'", "\"")
                self.metadata = dict(json.loads(json_str))
            
            assert set(self.metadata.keys()) == set(self.data.columns), "Loaded data and metadata does not match"
            for v in self.metadata.values():
                assert v in TrainingLogger.DATATYPES, f"Unknown datatype in metadata: {v}"
        except FileNotFoundError:
            try:
                os.makedirs(basename)
            except FileExistsError:
                if not overwrite:
                    q = input(f"The directory {basename} already exists. Delete existing? [y/N] ").lower()
                    overwrite = q in ("y", "yes", "j", "ja")
                if overwrite:
                    shutil.rmtree(basename)
                    os.makedirs(basename)
                else:
                    sys.exit(-1)
            self.data = pd.DataFrame()
            self.metadata = {}
        self.meta_changed = False
    
    def save(self):
        self.data.to_csv(os.path.join(self.basename, "data.csv"))
        if self.meta_changed:
            with open(os.path.join(self.basename, "data.meta"), "w") as f:
                f.write(str(self.metadata))
            self.meta_changed = False
    
    def add_scalar(self, name, value, iteration=None):
        if name not in self.data.columns:
            self.data[name] = None
            self.metadata[name] = "scalar"
            self.meta_changed = True

        assert self.metadata[name] == "scalar", f"Wrong datatype 'scalar' for column of type '{self.metadata[name]}'"

        if iteration is None:
            iteration = len(self.data.index)
        self.data.loc[iteration, name] = value
        self.save()
    
    def add_image(self, name, value, iteration=None):
        if name not in self.data.columns:
            self.data[name] = None
            self.metadata[name] = "img"
            self.meta_changed = True
            
        if iteration is None:
            iteration = len(self.data.index)
        assert self.metadata[name] == "img", f"Wrong datatype 'img' for column of type '{self.metadata[name]}''"
        
        if value.dtype == np.float32:
            value = (value * 255).astype('uint8')
        
        img_path = os.path.join(self.basename, f"{name}-{iteration}.png")
        i = Image.fromarray(value)
        i.save(img_path)
        i.close()

        self.data.loc[iteration, name] = img_path
        self.save()


class LogVisualizer:
    def __init__(self, logger):
        self.logger = logger
    
    def update_data(self):
        try:
            self.logger = TrainingLogger(self.logger.basename)
        except Exception as e:
            print("Could not update...")
            print(str(e))
    
    def show_graph(self, name, axes=None, ylim=None, xlim=None, **kwargs):
        assert name in self.logger.data.columns
        assert self.logger.metadata[name] == 'scalar'
        plt_data = self.logger.data[name].dropna()
        if axes is None:
            axes = plt.figure().gca()
        axes.plot(plt_data.index.values, plt_data.values, label=name)
        axes.legend()
        if ylim is not None:
            axes.set_ylim(ylim)
        if xlim is not None and isinstance(xlim, Iterable):
            axes.set_xlim(xlim)
        elif xlim is not None:
            axes.set_xlim((xlim, self.logger.data.index.max()))
        return axes
    
    def show_img(self, name, iteration, axes=None):
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
        i.close()
        return axes
    
    def show_scalars(self, names, subplots=True, axes=None, **kwargs):
        for name in names:
            axes = self.show_graph(name, axes, **kwargs)
            if subplots:
                axes = None
    
    def show_all_scalars(self, subplots=True, axes=None, **kwargs):
        for k, v in self.logger.metadata.items():
            if v == 'scalar':
                axes = self.show_graph(k, axes, **kwargs)
                if subplots:
                    axes = None
    
    def show_category_scalars(self, category, *args, single_axis = True, **kwargs):
        orig_axes = kwargs.pop('axes', None)
        axes = orig_axes
        for col in self.get_cols():
            if str(col).startswith(category):
                if not single_axis and orig_axes is None:
                    axes = None
                axes = self.show_graph(str(col), axes, *args, **kwargs)
        return axes
    
    def get_cols(self):
        return list(self.logger.data.columns)
    
    def get_non_null_index(self, col):
        return list(self.logger.data[col].dropna().index)