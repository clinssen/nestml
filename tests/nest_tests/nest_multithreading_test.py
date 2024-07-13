# -*- coding: utf-8 -*-
#
# nest_multithreading_test.py
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

import numpy as np
import os
import pytest

import nest

from pynestml.codegeneration.nest_tools import NESTTools
from pynestml.frontend.pynestml_frontend import generate_nest_target


@pytest.mark.parametrize("number_of_threads", [1, 2, 4])
class TestNestMultithreading:
    neuron_synapse_module = "nestml_stdp_module"
    neuron_synapse_neuron_model = "iaf_psc_exp_neuron_nestml__with_stdp_synapse_nestml"
    neuron_synapse_synapse_model = "stdp_synapse_nestml__with_iaf_psc_exp_neuron_nestml"

    neuron_module = "nestml_module"
    neuron_target = "/tmp/nestml-iaf-psc"
    neuron_model = "iaf_psc_exp_neuron__nestml"

    @pytest.fixture(autouse=True,
                    scope="session")
    def nestml_generate_target(self) -> None:
        """Generate the model code"""

        # Neuron-Synapse model
        neuron_path = os.path.join(
            os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "models",
                                          "neurons", "iaf_psc_exp_neuron.nestml")))
        synapse_path = os.path.join(
            os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "models",
                                          "synapses", "stdp_synapse.nestml")))
        generate_nest_target(input_path=[neuron_path, synapse_path],
                             logging_level="INFO",
                             module_name=self.neuron_synapse_module,
                             suffix="_nestml",
                             codegen_opts={"neuron_parent_class": "StructuralPlasticityNode",
                                           "neuron_parent_class_include": "structural_plasticity_node.h",
                                           "neuron_synapse_pairs": [{"neuron": "iaf_psc_exp_neuron",
                                                                     "synapse": "stdp_synapse",
                                                                     "post_ports": ["post_spikes"]}],
                                           "delay_variable": {"stdp_synapse": "d"},
                                           "weight_variable": {"stdp_synapse": "w"}})

        # Neuron model
        generate_nest_target(input_path=neuron_path,
                             logging_level="INFO",
                             module_name=self.neuron_module,
                             suffix="__nestml",
                             codegen_opts={"neuron_parent_class": "ArchivingNode",
                                           "neuron_parent_class_include": "archiving_node.h"})

    @pytest.mark.skipif(NESTTools.detect_nest_version().startswith("v2"),
                        reason="This test does not support NEST 2")
    def test_neuron_multithreading(self, number_of_threads: int) -> None:
        nest.ResetKernel()
        nest.resolution = 0.1
        nest.local_num_threads = number_of_threads

        try:
            nest.Install(self.neuron_module)
        except Exception:
            # ResetKernel() does not unload modules for NEST Simulator < v3.7; ignore exception if module is already loaded on earlier versions
            pass

        spike_times = np.array([2., 4., 7., 8., 12., 13., 19., 23., 24., 28., 29., 30., 33., 34.,
                                35., 36., 38., 40., 42., 46., 51., 53., 54., 55., 56., 59., 63., 64.,
                                65., 66., 68., 72., 73., 76., 79., 80., 83., 84., 86., 87., 90., 95.])
        sg = nest.Create("spike_generator",
                         params={"spike_times": spike_times})

        n = nest.Create(self.neuron_model, 5)
        nest.Connect(sg, n)

        multimeter = nest.Create("multimeter", params={"record_from": ["V_m"]})
        nest.Connect(multimeter, n)

        connections = nest.GetConnections()
        gid_post = np.unique(np.array(connections.get("target")))[0]
        nest.Simulate(100.)

        events = multimeter.get("events")
        v_m = events["V_m"]
        senders = events["senders"]
        v_m_sender = v_m[senders == gid_post]
        np.testing.assert_almost_equal(v_m_sender[-1], -69.97074345103812)

    @pytest.mark.skipif(NESTTools.detect_nest_version().startswith("v2"),
                        reason="This test does not support NEST 2")
    def test_neuron_synapse_multithreading(self, number_of_threads: int) -> None:
        pre_spike_times = np.array([2., 4., 7., 8., 12., 13., 19., 23., 24., 28., 29., 30., 33., 34.,
                                    35., 36., 38., 40., 42., 46., 51., 53., 54., 55., 56., 59., 63., 64.,
                                    65., 66., 68., 72., 73.])
        post_spike_times = np.array([4., 5., 6., 7., 10., 11., 12., 16., 17., 18., 19., 20., 22., 23.,
                                     25., 27., 29., 30., 31., 32., 34., 36., 37., 38., 39., 42., 44., 46.,
                                     48., 49., 50., 54., 56., 57., 59., 60., 61., 62., 67., 74.])

        nest.ResetKernel()
        nest.resolution = 0.1
        nest.local_num_threads = number_of_threads

        try:
            nest.Install(self.neuron_synapse_module)
        except Exception:
            # ResetKernel() does not unload modules for NEST Simulator < v3.7; ignore exception if module is already loaded on earlier versions
            pass

        wr = nest.Create("weight_recorder")
        nest.CopyModel(self.neuron_synapse_synapse_model, "stdp_nestml_rec",
                       {"weight_recorder": wr[0], "w": 1., "d": 1., "receptor_type": 0})

        # Spike generators
        pre_sg = nest.Create("spike_generator", 2,
                             params={"spike_times": pre_spike_times})
        post_sg = nest.Create("spike_generator", 2,
                              params={"spike_times": post_spike_times,
                                      "allow_offgrid_times": True})

        pre_neuron = nest.Create(self.neuron_synapse_neuron_model, 2)
        post_neuron = nest.Create(self.neuron_synapse_neuron_model, 2)
        sr_pre = nest.Create("spike_recorder")
        sr_post = nest.Create("spike_recorder")
        mm = nest.Create("multimeter", params={"record_from": ["V_m"]})

        nest.Connect(pre_sg, pre_neuron, "one_to_one", syn_spec={"delay": 1.})
        nest.Connect(post_sg, post_neuron, "one_to_one", syn_spec={"delay": 1., "weight": 9999.})
        nest.Connect(pre_neuron, post_neuron, "all_to_all", syn_spec={"synapse_model": "stdp_nestml_rec"})
        nest.Connect(mm, post_neuron)
        nest.Connect(pre_neuron, sr_pre)
        nest.Connect(post_neuron, sr_post)

        nest.Simulate(100.)

        connections = nest.GetConnections(synapse_model="stdp_nestml_rec")
        gid_post = np.unique(np.array(connections.get("target")))[0]
        events = mm.get("events")
        senders = events["senders"]
        V_m = events["V_m"]
        V_m_sender = V_m[senders == gid_post]
        np.testing.assert_almost_equal(V_m_sender[-1], -69.38156435373065)
