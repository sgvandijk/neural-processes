# Neural Processes

This is an implementation of Neural Processes, in Python using
TensorFlow. The original was ported from the [implementation in R by
Kaspar
Märtens](https://github.com/kasparmartens/NeuralProcesses). Since then
it has been updated for Tensorflow 2.

## Getting started

Python 3 is required, using a virtual environment is recommended.

If you want to run the accompanying notebook that explains Neural
Processes and shows examples, then install all requirements:

    pip install -r requirements.txt

If you don't have a global installation of Jupyter, then install that
as well into your environment:

    pip install jupyter

You can now run Jupyter and play around with the notebook:

    jupyter notebook

If you want to use the `neuralprocesses` code standalone, you only
need to install `tensorflow`, `tensorflow_probability` and `numpy` as
listed in `requirements.txt`. At the moment there is no support to
install the package using pip/setuptools, so either copy it over or
put it on `PYTHONPATH`.
