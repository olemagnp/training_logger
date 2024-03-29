from typing import Text
import pandas as pd
import numpy as np
import os, sys, shutil
import json
from PIL import Image


class TrainingLogger:
    """
    Class for logging the training of neural networks.

    The class is a simple wrapper around a pandas dataframe and a json-file.
    Data is saved in the dataframe, and saved as a CSV-file to disk.
    Metadata is saved in the json-file.
    Images are saved in the root folder
    """

    DATATYPES = set({"scalar", "img", "text"})

    def __init__(self, basename, overwrite=False, save_freq=50):
        """
        Create a new logger, which logs to :code:`basename`.

        The logger can either work with existing logs. If so, the user will be asked to
        confirm, as this could possibly overwrite existing data. You can circumwent this
        check by passing :code:`overwrite=True` to the initializer.

        The initializer further makes sure the logdir exists, and initializes the internal
        dataframe and metadata dict.

        :param basename: The directory to save logs, metadata, and images to.
        :param overwrite: If :code:`True`, no check is made to confirm that you wish to work with existing logs.
        :param save_freq: The number of changes that must be made before data is saved to file.
        """
        self.basename = basename
        try:
            self.data = pd.read_csv(os.path.join(basename, "data.csv"), index_col=0)
            if not overwrite:
                q = input(
                    f"The directory {basename} already exists. Continue (this may overwrite old data)? [y/N] "
                ).lower()
                if q not in ("y", "yes", "j", "ja"):
                    print("Exiting due to existing file")
                    sys.exit(-1)
            with open(os.path.join(basename, "data.meta"), "r") as f:
                json_str = f.read().replace("'", '"')
                self.metadata = dict(json.loads(json_str))

            assert set(self.metadata.keys()) == set(
                self.data.columns
            ), "Loaded data and metadata does not match"
            for v in self.metadata.values():
                assert (
                    v in TrainingLogger.DATATYPES
                ), f"Unknown datatype in metadata: {v}"
        except FileNotFoundError:
            try:
                os.makedirs(basename)
            except FileExistsError:
                if not overwrite:
                    q = input(
                        f"The directory {basename} already exists. Delete existing? [y/N] "
                    ).lower()
                    overwrite = q in ("y", "yes", "j", "ja")
                if overwrite:
                    shutil.rmtree(basename)
                    os.makedirs(basename)
                else:
                    sys.exit(-1)
            self.data = pd.DataFrame()
            self.metadata = {}
        self.meta_changed = False
        self.save_freq = save_freq
        self.save_calls = 0
        self.prefix = ""
        self.postfix = ""

    def flush(self):
        """
        Force write all new data to file.
        """
        self.save(True)

    def save(self, force=False):
        """
        Save the content of the logger.

        The method always saves the dataframe to a csv at :nocode:`<basename>/data.csv`,
        but only saves the metadata to :nocode:`<basename>/data.meta` if the metadata
        has changed (i.e. if a new column has been added).
        """
        self.save_calls += 1
        if force or self.save_calls >= self.save_freq:
            self.data.to_csv(os.path.join(self.basename, "data.csv"))
            if self.meta_changed:
                with open(os.path.join(self.basename, "data.meta"), "w") as f:
                    f.write(str(self.metadata))
                self.meta_changed = False
            self.save_calls = 0

    def _get_name(self, name):
        name = self.prefix + name + self.postfix
        return name

    def add_to_prefix(self, name: Text):
        self.prefix = self.prefix + name

    def remove_from_prefix(self, name: Text):
        if not self.prefix.endswith(name):
            raise ValueError("Can only remove last part of prefix.")
        self.prefix = self.prefix[: -len(name)]

    def add_to_postfix(self, name: Text):
        self.postfix = self.postfix + name

    def remove_from_postfix(self, name: Text):
        if not self.postfix.endswith(name):
            raise ValueError("Can only remove last part of postfix.")
        self.postfix = self.postfix[: -len(name)]
        self.postfix = None

    def __insert_scalar(self, name, value, iteration=None):
        name = self._get_name(name)

        if name not in self.data.columns:
            self.data[name] = None
            self.metadata[name] = "scalar"
            self.meta_changed = True

        assert (
            self.metadata[name] == "scalar"
        ), f"Wrong datatype 'scalar' for column of type '{self.metadata[name]}'"

        if iteration is None:
            iteration = len(self.data.index)
        self.data.loc[iteration, name] = value

    def __insert_text(self, name, value, iteration=None):
        name = self._get_name(name)
        if name not in self.data.columns:
            self.data[name] = None
            self.metadata[name] = "text"
            self.meta_changed = True

        assert (
            self.metadata[name] == "text"
        ), f"Wrong datatype 'text' for column of type '{self.metadata[name]}'"

        if iteration is None:
            iteration = len(self.data.index)
        self.data.loc[iteration, name] = value

    def add_scalar(self, name, value, iteration=None):
        """
        Add a scalar value to the log.

        If the given scalar name does not exist yet, a new
        column is added to the dataframe and metadata.

        The method saves the data to file after adding.

        :param name: Name of the value, used to group and display the data.
        :param value: Scalar value to add. Note that this should be a number, not a Tensor or Array,
                    for it to work properly with the visualization. However, no check is made against
                    this, meaning you are free to save whatever you wish.
        :param iteration: Iteration to save this image to. Used for displaying the data. If None, :code:`iteration = len(self.data.index)`
        """
        self.__insert_scalar(name, value, iteration)
        self.save()

    def add_text(self, name, value, iteration=None):
        """
        Add a text value to the log.

        If the given text name does not exist yet, a new
        column is added to the dataframe and metadata.

        The method saves the data to file after adding.

        :param name: Name of the value, used to group and display the data.
        :param value: Text value to add. No check is made against the type of this
                    parameter, meaning you are free to save whatever you wish.
        :param iteration: Iteration to save this image to. Used for displaying the data.
                    If None, :code:`iteration = len(self.data.index)`
        """
        self.__insert_text(name, value, iteration)
        self.save()

    def add_image(self, name, value, iteration=None):
        """
        Add an image to the log.

        If the given image name does not exist yet, a new
        column is added to the dataframe and metadata.

        The method saves the data to file after adding.

        :param name: Name of the value, used to group and display the data.
        :param value: Image to be added. This should be in the form of a :class:`numpy.ndarray` or similar. Type can be either :obj:`numpy.float32` (with values in the range :math:`[0, 1)` ) or :obj:`numpy.uint8` (with values in the range :math:`[0,256]` ).
        :param iteration: Iteration to save this image to. Used for displaying the data. If :code:`None`, :code:`iteration = len(self.data.index)`
        """
        name = self._get_name(name)
        if name not in self.data.columns:
            self.data[name] = None
            self.metadata[name] = "img"
            self.meta_changed = True

        if iteration is None:
            iteration = len(self.data.index)
        assert (
            self.metadata[name] == "img"
        ), f"Wrong datatype 'img' for column of type '{self.metadata[name]}''"

        if value.dtype == np.float32:
            value = (value * 255).astype("uint8")

        img_path = os.path.join(self.basename, f"{name}-{iteration}.png")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        i = Image.fromarray(value)
        i.save(img_path)
        i.close()

        self.data.loc[iteration, name] = img_path
        self.save()

    def add_scalars(self, pre: str = "", post: str = "", iteration=None, **keyvals):
        """
        Add multiple scalars, given as key-value pairs.

        If a given scalar name does not exist yet, a new
        column is added to the dataframe and metadata.

        The method saves the data to file after adding.

        :param keyvals: Name and values to add. Note that the values should be numbers, not Tensors or Arrays,
                    for them to work properly with the visualization. However, no check is made against
                    this, meaning you are free to save whatever you wish.
        :param pre: String to add before every name
        :param post: String to add after every name
        :param iteration: Iteration to save this image to. Used for displaying the data. If None, :code:`iteration = len(self.data.index)`
        """
        old_pre, old_post = self.prefix, self.postfix

        self.prefix = self.prefix + pre
        self.postfix = self.postfix + post

        if iteration is None:
            iteration = len(self.data.index)

        for name, val in keyvals.items():
            self.__insert_scalar(name, val, iteration)
        self.save()

        self.prefix, self.postfix = old_pre, old_post