""" Project initialization and common objects. """

import logging
import os
from pathlib import Path
import re
import sys
import pickle
from sklearn.externals import joblib
import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)

workdir = Path(os.getenv('WORKDIR', '.'))

cachedir = workdir / 'cache'
cachedir.mkdir(parents=True, exist_ok=True)

#: Sets the collision systems for the entire project,
#: where each system is a string of the form
#: ``'<projectile 1><projectile 2><beam energy in GeV>'``,
#: such as ``'PbPb2760'``, ``'AuAu200'``, ``'pPb5020'``.
#: Even if the project uses only a single system,
#: this should still be a list of one system string.
systems = ['PbPb5020']

observables=['obs:R_AA-sys:PbPb5020-pT:10.8',
            'obs:R_AA-sys:PbPb5020-pT:13.2',
            'obs:R_AA-sys:PbPb5020-pT:16.8',
            'obs:R_AA-sys:PbPb5020-pT:21.6',
            'obs:R_AA-sys:PbPb5020-pT:32.0']


#: Design attribute. This is a list of 
#: strings describing the inputs.
#: The default is for the example data.
keys = ['lambda_jet','alpha_s'] #labels in words

#: Design attribute. This is a list of input
#: labels in LaTeX for plotting.
#: The default is for the example data. 
labels = [r'lambda_jet',r'alpha_s'] #labels in LaTeX

#: Design attribute. This is list of tuples of 
#: (min,max) for each design input.
#: The default is for the example data.
ranges = [(0.01,0.3),(0.05,0.35)]

#: Design array to use - should be a numpy array.
#: Keep at None generate a Latin Hypercube with above (specified) range.
#: Design array for example is commented under default.
#design_array = None
design_array = pickle.load((cachedir / 'lhs/design_s.p').open('rb'))
#print(design_array)
#: Dictionary of the model output.
#: Form MUST be data_list[system][observable][subobservable][{'Y': ,'x': }].
#:     'Y' is an (n x p) numpy array of the output.
#:
#:     'x' is a (1 x p) numpy array of numeric index of columns of Y (if exists). In the example data, x is p_T. 
#: This MUST be changed from None - no built-in default exists. Uncomment the line below default for example.
data_list = np.array([[0.668, 0.71 , 0.764, 0.809, 0.832],
       [0.338, 0.382, 0.443, 0.507, 0.558],
       [0.21 , 0.246, 0.304, 0.358, 0.405],
       [0.327, 0.363, 0.42 , 0.476, 0.509],
       [0.249, 0.291, 0.347, 0.396, 0.443],
       [0.275, 0.31 , 0.367, 0.418, 0.46],
       [0.314, 0.353, 0.419, 0.463, 0.508],
       [0.18 , 0.214, 0.266, 0.31 , 0.357],
       [0.405, 0.448, 0.521, 0.576, 0.616],
       [0.243, 0.28 , 0.326, 0.384, 0.423],
       [0.424, 0.457, 0.516, 0.568, 0.612],
       [0.174, 0.205, 0.254, 0.315, 0.353],
       [0.365, 0.401, 0.463, 0.521, 0.566],
       [0.429, 0.473, 0.531, 0.574, 0.619],
       [0.231, 0.26 , 0.317, 0.369, 0.41],
       [0.529, 0.565, 0.627, 0.685, 0.711],
       [0.303, 0.339, 0.398, 0.457, 0.494],
       [0.457, 0.491, 0.557, 0.61 , 0.656],
       [0.455, 0.499, 0.577, 0.633, 0.673],
       [0.542, 0.578, 0.647, 0.709, 0.743],
       [0.381, 0.417, 0.471, 0.524, 0.561],
       [0.128, 0.155, 0.195, 0.242, 0.283],
       [0.421, 0.455, 0.512, 0.568, 0.603],
       [0.301, 0.343, 0.411, 0.478, 0.529]])
#data_list = pickle.load((cachedir / 'model/main/full_data_dict.p').open('rb'))
#: Dictionary for the model validation output
#: Must be the same for as the model output dictionary
#data_list_val = pickle.load((cachedir / 'model/validation/data_dict_val.p').open('rb'))
data_list_val = None

#: Dictionary of the experimental data.
#: Form MUST be exp_data_list[system][observable][subobservable][{'y':,'x':,'yerr':{'stat':,'sys'}}].
#:      'y' is a (1 x p) numpy array of experimental data.
#:
#:      'x' is a (1 x p) numpy array of numeric index of columns of Y (if exists). In the example data, x is p_T.
#:
#:      'yerr' is a dictionary with keys 'stat' and 'sys'.
#:
#:      'stat' is a (1 x p) array of statistical errors.
#:
#:      'sys' is a (1 x p) array of systematic errors.
#: This MUST be changed from None - no built-in default exists. Uncomment the line below default for example.
exp_data_param_list=np.array([[10.8],[13.2],[16.8],[21.6],[32.0]])
exp_data_list = np.array([[0.457, 0.0409, 0.00150],
[0.506, 0.0454, 0.0021],
[0.557, 0.0477, 0.0041],
[0.6, 0.0513, 0.0068],
[0.644, 0.0551, 0.0092]])
#exp_data_list = pickle.load((cachedir / 'hepdata/data_list_exp.p').open('rb'))

#: Experimental covariance matrix.
#: Set exp_cov = None to have the script estimate the covariance matrix.
#: Example commented below default.
exp_cov = np.array([[0.00117728, 0.00140226, 0.00155336, 0.00162377, 0.001733],
 [0.00140226, 0.00167506, 0.00185512, 0.00194179, 0.00207503],
 [0.00155336, 0.00185512, 0.00206557, 0.00216207, 0.00231496],
 [0.00162377, 0.00194179, 0.00216207, 0.0022921,  0.00244304],
 [0.001733,   0.00207503, 0.00231496, 0.00244304, 0.00267793]])
#exp_cov = pickle.load((cachedir / 'hepdata/cov_exp_pbpb5020_30_50.p').open('rb'))

#: Observables to emulate as a list of 2-tuples
#: ``(obs, [list of subobs])``.
#observables = [('R_AA',[None])]

def parse_system(system):
    """
    Parse a system string into a pair of projectiles and a beam energy.

    """
    match = re.fullmatch('([A-Z]?[a-z])([A-Z]?[a-z])([0-9]+)', system)
    return match.group(1, 2), int(match.group(3))


class lazydict(dict):
    """
    A dict that populates itself on demand by calling a unary function.

    """
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __missing__(self, key):
        self[key] = value = self.function(key, *self.args, **self.kwargs)
        return value
