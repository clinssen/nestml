#
# saturating_integration_test.py
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

import nest
import numpy as np
import os
import unittest
from pynestml.frontend.pynestml_frontend import to_nest, install_nest

try:
    import matplotlib
    import matplotlib.pyplot as plt
    TEST_PLOTS = True
except:
    TEST_PLOTS = False


class NestSaturatingIntegrationTest(unittest.TestCase):

    def test_saturating_integration(self):
        input_path = os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), "resources", "saturating_integration.nestml")))
        nest_path = "/home/archels/nest-simulator-build"
        target_path = 'target'
        logging_level = 'INFO'
        module_name = 'nestmlmodule'
        store_log = False
        suffix = '_nestml'
        dev = True
        to_nest(input_path, target_path, logging_level, module_name, store_log, suffix, dev)
        install_nest(target_path, nest_path)
        nest.set_verbosity("M_ALL")

        nest.ResetKernel()
        nest.Install(module_name)

        # network construction

        neuron = nest.Create("saturating_integration_nestml")

        sg = nest.Create("spike_generator", params={"spike_times": np.linspace(10, 100, 10)})
        nest.Connect(sg, neuron, syn_spec={"weight": 1., "delay": .1})

        mm = nest.Create('multimeter', params={'record_from': ['I_syn', 'k__X__spikes'], 'interval': 0.1})
        nest.Connect(mm, neuron)

        #vm_1 = nest.Create('voltmeter')
        #nest.Connect(vm_1, neuron)

        # simulate

        nest.Simulate(125.)

        # analysis
        mm = nest.GetStatus(mm)[0]["events"]
        V_m_timevec = mm["times"]
        V_m = mm["I_syn"]
        #MAX_ABS_ERROR = 1E-6
        #print("Final V_m = " + str(V_m[-1]))
        #assert abs(V_m[-1] - -72.89041451202348) < MAX_ABS_ERROR

        if TEST_PLOTS:

            fig, ax = plt.subplots(nrows=2)

            ax[0].plot(V_m_timevec, V_m, label="I_syn")
            ax[0].set_ylabel("current")

            ax[1].plot(mm["times"], mm["k__X__spikes"], label="k__X__spikes")
            ax[1].set_ylabel("current")

            for _ax in ax:
                _ax.legend(loc="upper right")
                #_ax.set_xlim(0., 125.)
                _ax.grid(True)

            for _ax in ax[:-1]:
                _ax.set_xticklabels([])

            ax[-1].set_xlabel("time")

            plt.savefig("/tmp/saturating_integration.png")
