from sphinx.setup_command import BuildDoc
from setuptools import setup

cmdclass = {'build_sphinx': BuildDoc}

name = "training-logger"
version = "0.1"
release = "0.1.0"

setup(
    name=name,
    version=release,
    description="Simple utility to log PyTorch training to CSV",
    author="Ole-Magnus Pedersen",
    url="https://github.com/olemagnp/training_logger",
    author_email="pedersen.olemagnus@gmail.com",
    license="MIT",
    packages=["training_logger"],
    zip_safe=False,
    cmdclass=cmdclass,
    command_options={
        'project': ('setup.py', name),
        'version': ('setup.py', version),
        'release': ('setup.py', release),
        'source_dir': ('setup.py', './doc/source'),
        'build-dir': ('setup.py', './doc/build')
    },
    install_requires=(
        'pillow',
        'numpy',
        'pandas',
        'matplotlib',
        'sphinx',
        'sphinx_rtd_theme'
    )
)