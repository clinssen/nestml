# -*- coding: utf-8 -*-
#
# stdp_synapse_test.py
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

from typing import Sequence

import numpy as np
import os
import pytest

import nest

from pynestml.codegeneration.nest_tools import NESTTools
from pynestml.frontend.pynestml_frontend import generate_nest_target

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.ticker
    import matplotlib.pyplot as plt
    TEST_PLOTS = True
except Exception:
    TEST_PLOTS = False

sim_mdl = True
sim_ref = True


class TestNestSTDPSynapse:

    neuron_model_name = "iaf_psc_exp_multisyn_nestml__with_stdp_nestml"
    synapse_model_name = "stdp_nestml__with_iaf_psc_exp_multisyn_nestml"

    @pytest.fixture(autouse=True,
                    scope="module")
    def generate_model_code(self):
        """Generate the model code"""
        jit_codegen_opts = {"neuron_synapse_pairs": [{"neuron": "iaf_psc_exp_multisyn",
                                                      "synapse": "stdp",
                                                      "post_ports": ["post_spikes"]}]}

        files = [os.path.join("models", "neurons", "iaf_psc_exp_multisyn.nestml"),
                 os.path.join("models", "synapses", "stdp_synapse.nestml")]
        input_path = [os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.join(
            os.pardir, os.pardir, s))) for s in files]
        generate_nest_target(input_path=input_path,
                             target_path="/tmp/nestml-jit",
                             logging_level="INFO",
                             module_name="nestml_jit_module",
                             suffix="_nestml",
                             codegen_opts=jit_codegen_opts)

    def test_nest_stdp_synapse(self):

        nest.Install("nestml_jit_module")
        nest.set_verbosity("M_ALL")

        nest.ResetKernel()
        pre_neuron = nest.Create("poisson_generator", params={"rate": 20.})
        post_neuron = nest.Create(self.neuron_model_name)
        nest.Connect(pre_neuron, post_neuron, "all_to_all", syn_spec={"synapse_model": self.synapse_model_name, "receptor_type": 2})
        nest.Simulate(100.)

        nest.ResetKernel()
        pre_neuron = nest.Create("poisson_generator", params={"rate": 20.})
        post_neuron = nest.Create(self.neuron_model_name)
        nest.Connect(pre_neuron, post_neuron, "all_to_all", syn_spec={"synapse_model": self.synapse_model_name, "receptor_type": 3})
        nest.Simulate(100.)
