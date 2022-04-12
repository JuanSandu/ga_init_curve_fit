"""
Code to identify a system from the given data using the GA-improved curve-fit
function from Scipy.

Author: Juan Sandubete Lopez
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.stats
from scipy import signal as sig
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.special import erf

from genetic_fit_curve import GA_fit_curve

dirname = os.path.dirname(__file__)


class TF_sys():
    def __init__(self):
        self.tf = None
        self.inputs = None
        self.t = None
        self.k = 6.6
        self.wn = 5.5
        self.delta = 0.5

    def second_order_mdl(self, inputs, k, wn, delta):
        self.tf = sig.TransferFunction(k*(wn**2), [1, 2*delta*wn, wn**2])
        to, yo, xo = sig.lsim2(self.tf, U=inputs, T=self.t)
        return yo

    def simulate(self, timeline, inputs, noise_dev=0.5):
        # self.inputs = inputs
        self.t = timeline
        yo = self.second_order_mdl(inputs, self.k, self.wn, self.delta)
        noise = np.random.normal(0, noise_dev, len(timeline))
        return np.add(yo, noise)


def run():
    # Generate the data
    sys = TF_sys()
    # For 100 data points
    timeline = list(np.linspace(0, 5, 100))
    control_sig = [0]*10 + [10]*90
    out_sig = sys.simulate(timeline, control_sig)
    # plt.plot(timeline, out_sig, 'b')
    # plt.plot(timeline, control_sig, 'r')
    # plt.show()

    # Configure and run the identification
    ga_fit = GA_fit_curve( [1.0, 1.0, 1.0], sys.second_order_mdl,
                           control_sig, out_sig, debug_print=True )
    ga_fit.load_config("config_tf2.json")
    ga_fit.run()

    # Return the values
    k, wn, delta = ga_fit.get_best_params()
    print("Parameters: k: {}, wn: {}, delta: {}".format(k, wn, delta))


run()
