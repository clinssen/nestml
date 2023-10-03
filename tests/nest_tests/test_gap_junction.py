# -*- coding: utf-8 -*-
#
# test_gap_junction.py
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
import scipy

import nest

from pynestml.frontend.pynestml_frontend import generate_nest_target

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.ticker
    import matplotlib.pyplot as plt
    TEST_PLOTS = True
except Exception:
    TEST_PLOTS = False


class TestGapJunction:
    r"""Test code generation and perform simulations and numerical checks for gap junction support in linear and non-linear neuron models"""

    @pytest.fixture(params=["iaf_psc_exp", "aeif_cond_exp"])
    def generate_code(self, request):
        neuron_model: str = request.param

        codegen_opts = {"gap_junctions": {"enable": True,
                                          "gap_current_port": "I_stim",
                                          "membrane_potential_variable": "V_m"}}

        files = [os.path.join("models", "neurons", neuron_model + ".nestml")]
        input_path = [os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.join(os.pardir, os.pardir, s))) for s in files]
        generate_nest_target(input_path=input_path,
                             logging_level="DEBUG",
                             module_name="nestml_gap_module",
                             suffix="_nestml",
                             codegen_opts=codegen_opts)

        nest.Install("nestml_gap_module")

        return neuron_model

    @pytest.mark.parametrize("wfr_interpolation_order", [0, 1, 3])
    def test_gap_junction_effect_on_membrane_potential(self, generate_code, wfr_interpolation_order: int):
        neuron_model = generate_code    # hacky because of pytest limitations -- loop over neuron models

        resolution = .1   # [ms]
        sim_time = 100.   # [ms]
        pre_spike_times = [1., 16., 31.]    # [ms]

        nest.set_verbosity("M_ALL")
        nest.ResetKernel()

        nest.resolution = resolution
        nest.wfr_comm_interval = 2.         # [ms]
        nest.wfr_interpolation_order = wfr_interpolation_order

        pre_neuron = nest.Create(neuron_model + "_nestml")
        post_neuron = nest.Create(neuron_model + "_nestml")

        pre_sg = nest.Create("spike_generator",
                             params={"spike_times": pre_spike_times})
        pre_parrot = nest.Create("parrot_neuron")
        nest.Connect(pre_sg, pre_parrot)
        nest.Connect(pre_parrot, pre_neuron, syn_spec={"weight": 999.})

        nest.Connect(pre_neuron,
                     post_neuron,
                     conn_spec={"rule": "one_to_one", "make_symmetric": True},
                     syn_spec={"synapse_model": "gap_junction"})

        mm_pre = nest.Create("multimeter", params={"record_from": ["V_m"]})
        nest.Connect(mm_pre, pre_neuron)

        mm_post = nest.Create("multimeter", params={"record_from": ["V_m"]})
        nest.Connect(mm_post, post_neuron)

        nest.Simulate(sim_time)

        # plot
        if TEST_PLOTS:
            fig, ax = plt.subplots(nrows=2)
            ax1, ax2 = ax

            timevec = nest.GetStatus(mm_pre, "events")[0]["times"]
            V_m = nest.GetStatus(mm_pre, "events")[0]["V_m"]
            ax1.plot(timevec, V_m)
            ax1.set_ylabel("V_m pre")

            timevec = nest.GetStatus(mm_post, "events")[0]["times"]
            V_m = nest.GetStatus(mm_post, "events")[0]["V_m"]
            ax2.plot(timevec, V_m)
            ax2.set_ylabel("V_m post")

            for _ax in ax:
                _ax.grid(which="major", axis="both")
                _ax.grid(which="minor", axis="x", linestyle=":", alpha=.4)
                _ax.set_xlim(0., sim_time)
                _ax.legend()

            fig.suptitle("wfr interpolation order: " + str(wfr_interpolation_order))
            fig.savefig("/tmp/gap_junction_test_[neuron_model=" + neuron_model + "]_[wfr_order=" + str(wfr_interpolation_order) + "].png", dpi=300)

        V_m_log = nest.GetStatus(mm_post, "events")[0]["V_m"]

        # assert that gap currents bring the neuron at least 0.5% closer to threshold
        assert np.amax(V_m_log) > .995 * post_neuron.E_L + .005 * post_neuron.V_th

        # assert that there are n_pre_spikes peaks in V_m
        assert len(scipy.signal.find_peaks(V_m_log)[0]) >= len(pre_sg.spike_times)

        pre_neuron.E_L = -70.
        post_neuron.E_L = -80.

        nest.Simulate(100.)

        np.testing.assert_allclose(pre_neuron.V_m, -99.)
        np.testing.assert_allclose(post_neuron.V_m, -99.)
