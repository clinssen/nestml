# -*- coding: utf-8 -*-
#
# wb_cond_exp_test.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

try:
    import matplotlib as mpl
    mpl.use("agg")
    import matplotlib.pyplot as plt
    TEST_PLOTS = True
except BaseException:
    TEST_PLOTS = False


import os
import unittest
import numpy as np

import nest

from pynestml.codegeneration.nest_tools import NESTTools
from pynestml.frontend.pynestml_frontend import generate_nest_target


class NestWBCondExpTest(unittest.TestCase):

    def test_wb_cond_exp(self):

        if not os.path.exists("target"):
            os.makedirs("target")

        input_path = os.path.join(os.path.realpath(os.path.join(
            os.path.dirname(__file__), os.pardir, os.pardir, "models", "neurons", "wb_cond_exp.nestml")))
        target_path = "target"
        module_name = "nestmlmodule"
        suffix = "_nestml"
        nest_version = NESTTools.detect_nest_version()

        generate_nest_target(input_path,
                             target_path=target_path,
                             logging_level="INFO",
                             suffix=suffix,
                             module_name=module_name)

        nest.Install(module_name)
        model = "wb_cond_exp_nestml"

        dt = 0.01
        t_simulation = 1000.0
        nest.SetKernelStatus({"resolution": dt})

        neuron = nest.Create(model)

        nest.SetStatus(neuron, {"I_e": 75.0})
        multimeter = nest.Create("multimeter")
        nest.SetStatus(multimeter, {"record_from": ["V_m"],
                                    "interval": dt})
        if nest_version.startswith("v2"):
            spike_recorder = nest.Create("spike_detector")
        else:
            spike_recorder = nest.Create("spike_recorder")
        nest.Connect(multimeter, neuron)
        nest.Connect(neuron, spike_recorder)
        nest.Simulate(t_simulation)

        dmm = nest.GetStatus(multimeter)[0]
        Voltages = dmm["events"]["V_m"]
        tv = dmm["events"]["times"]
        dSD = nest.GetStatus(spike_recorder, keys="events")[0]
        spikes = dSD["senders"]
        ts = dSD["times"]

        firing_rate = len(spikes) / t_simulation * 1000
        print("firing rate is ", firing_rate)
        expected_value = np.abs(firing_rate - 50)
        tolerance_value = 5  # Hz

        if TEST_PLOTS:
            fig, ax = plt.subplots(2, figsize=(8, 6), sharex=True)
            ax[0].plot(tv, Voltages, lw=2, color="k")
            ax[1].plot(ts, spikes, "ko")
            ax[1].set_xlabel("Time [ms]")
            ax[1].set_xlim(0, t_simulation)
            ax[1].set_ylabel("Spikes")
            ax[0].set_ylabel("v [ms]")
            ax[0].set_ylim(-100, 50)

            for i in ts:
                ax[0].axvline(x=i, lw=1., ls="--", color="gray")

            plt.savefig("wb_cond_exp.png")

        self.assertLessEqual(expected_value, tolerance_value)


if __name__ == "__main__":
    unittest.main()
