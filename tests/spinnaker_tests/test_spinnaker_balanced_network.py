# -*- coding: utf-8 -*-
#
# test_spinnaker_balanced_network.py
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

from fileinput import filename
import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
import csv
import pyNN.spiNNaker as p
from pyNN.utility.plotting import Figure, Panel

from pynestml.frontend.pynestml_frontend import generate_spinnaker_target
from pyNN.random import RandomDistribution

from datetime import datetime
import time

from analyze_sample_profile import analyze_latest_profiles



# get results before test after plot codes  first plot and then test
# next stdp synapses instead of static ones in here
# try built in neuron with nestml synapse ( stdp synapse)
# dont take too much time
# important parameters learning rate or lambda

# we need to use static and plastic synapses. it tried to act like a static synapse but it is plastic synapse. 
# it is a plastic synapse with learning rate 0.0.


# Keep False until the built-in STDP implementation
# is confirmed.
use_builtin_stdp = True


def compute_cv(spike_train):
    """
    Compute the coefficient of variation (CV) for a single spike train.

    Parameters:
    spike_train (list or numpy array): Timestamps of spikes in the spike train.

    Returns:
    float: Coefficient of variation (CV) of the inter-spike intervals.
    """
    # Calculate inter-spike intervals (ISI)
    if len(spike_train) < 3:
        return np.nan  

    isi = np.diff(spike_train)

    # Calculate mean and standard deviation of ISI
    mean_isi = np.mean(isi)

    if mean_isi == 0:
        return np.nan  # Avoid division by zero

    std_isi = np.std(isi)

    # Calculate coefficient of variation
    cv = std_isi / mean_isi

    return cv


def compute_cv_for_neurons(spike_trains):
    """
    Compute the average coefficient of variation (CV) for a population of neurons.
    """
    cvs = []

    for spike_train in spike_trains:

        cv = compute_cv(spike_train)

        if not np.isnan(cv):
            cvs.append(cv)

    if not cvs:
        return np.nan  

    return np.mean(cvs)


def compute_average_firing_rate(spike_trains, n_neurons, t_sim):
    """
    Compute the population-average firing rate in Hz.
    """
    # Calculate the number of spikes
    total_spikes = sum(len(spike_train) for spike_train in spike_trains)
                       
    # Convert ms to seconds
    sim_time_seconds = t_sim / 1000.0

    # Calculate average firing rate
    avg_firing_rate = total_spikes / (n_neurons * sim_time_seconds)

    return avg_firing_rate


def compare_results(reference, results):
    """
    Compare NESTML results with the reference implementation.
    """
    comparisons={}

    for name, result in results.items():
        if name == "reference":
            continue

        if result is None:
            continue

        comparisons[name] = {
            "exc_relative_difference": (abs(result["exc_firing_rate"] - reference["exc_firing_rate"]) / reference["exc_firing_rate"]),
            "inh_relative_difference": (abs(result["inh_firing_rate"] - reference["inh_firing_rate"]) / reference["inh_firing_rate"]),
            "cv_absolute_difference": abs(result["cv"] - reference["cv"]),
            "execution_time_difference": abs(result["execution_time"] - reference["execution_time"])
        }


    return comparisons

def print_results(results, comparisons):

    print("\n===================================")
    print("RESULTS")
    print("===================================")

    for name, result in results.items():

        if result is None:
            print(f"\n{name}: SKIPPED")
            continue

        print(f"\n{name}")

        print(f"Exc. firing rate = {result['exc_firing_rate']:.3f} Hz")
        print(f"Inh. firing rate = {result['inh_firing_rate']:.3f} Hz")
        print(f"CV = {result['cv']:.3f}")

        print(f"Execution time = {result['execution_time']:.3f} s")

        print(f"E->E weight = {result['mean_w_ee']:.3f}")
        print(f"E->I weight = {result['mean_w_ei']:.3f}")
        print(f"I->E weight = {result['mean_w_ie']:.3f}")
        print(f"I->I weight = {result['mean_w_ii']:.3f}")
        
    if comparisons:
        print("\n===================================")
        print("REFERENCE COMPARISON")
        print("===================================")

        for name, comparison in comparisons.items():

            print(f"\nreference vs {name}")

            print(f"Exc. relative difference = {comparison['exc_relative_difference']:.2%}")
            print(f"Inh. relative difference = {comparison['inh_relative_difference']:.2%}")
            print(f"  CV absolute difference = {comparison['cv_absolute_difference']:.3f}")

            print(f"  Execution time difference = {comparison['execution_time_difference']:.3f} s")


def save_results_to_csv(results):

    filename = "stdp_balanced_networks_results.csv"


    fieldnames = ["timestamp", "implementation", "N", "g", "p_conn", "rate_ext_input", "neurons_per_core", "t_sim",
        "exc_firing_rate", "inh_firing_rate", "cv", "execution_time",
        "mean_w_ee", "mean_w_ei", "mean_w_ie", "mean_w_ii"
    ]

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists: 
            writer.writeheader()


        for name, result in results.items():

            if result is None:
                continue

            row = {
                "timestamp": result["timestamp"],
                "implementation": result["implementation"],
                "N": result["n_neurons"],
                "g": result["g"],
                "p_conn": result["p_conn"],
                "rate_ext_input": result["rate_ext_input"],
                "neurons_per_core": result["neurons_per_core"],
                "t_sim": result["t_sim"],
                "exc_firing_rate": result["exc_firing_rate"],
                "inh_firing_rate": result["inh_firing_rate"],
                "cv": result["cv"],
                        
                "execution_time": result["execution_time"],

                "mean_w_ee": result["mean_w_ee"],
                "mean_w_ei": result["mean_w_ei"],
                "mean_w_ie": result["mean_w_ie"],
                "mean_w_ii": result["mean_w_ii"]
            }
    

            writer.writerow(row)

def save_comparisons_to_csv(reference, comparisons):

    filename = "stdp_balanced_networks_comparisons.csv"

    fieldnames = ["timestamp", "implementation",
                  "exc_relative_difference", "inh_relative_difference", "cv_absolute_difference",
                  "execution_time_difference"]

    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists: 
            writer.writeheader()

        for name, comparison in comparisons.items():

            row = {
                "timestamp": reference["timestamp"],
                "implementation": name,
                "exc_relative_difference": comparison["exc_relative_difference"],
                "inh_relative_difference": comparison["inh_relative_difference"],
                "cv_absolute_difference": comparison["cv_absolute_difference"],
                "execution_time_difference": comparison["execution_time_difference"]
            }

            writer.writerow(row)


def plot_membrane_potential(results):
    """
    Plot the membrane potential of a neuron and save the figure.
    """
    fig, ax = plt.subplots(nrows=1, dpi=300)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Membrane potential [mV]")
    ax.set_title(f"\nBalanced Network - {results['implementation']} | N={results['n_neurons']}, g={results['g']}, neurons/core={results['neurons_per_core']} \n"
                 f"p_conn={results['p_conn']}, input={results['rate_ext_input']} Hz |\n")

    ax.plot(results['v_neuron'].segments[0].filter(name=results['membranePot'])[0])

    plt.savefig(
        f"spinnaker_balanced_network_V_m__ {results['implementation']}_{results['timestamp']}.png"
    )
    plt.close("all")


def plot_raster(results):
    """
    Plot the raster of spikes for excitatory and inhibitory populations and save the figure.
    """

    Figure(Panel(results['pop_exc_spikes'].segments[0].spiketrains,
                 xlabel="Time/ms",
                 xticks=True,
                 yticks=True,
                 markersize=5,
                 xlim=(0, results["t_sim"])),
               # raster plot of the presynaptic neuron spike times
           Panel(results['pop_inh_spikes'].segments[0].spiketrains,
                 xlabel="Time/ms",
                 xticks=True,
                 yticks=True,
                 markersize=5,
                 xlim=(0, results["t_sim"])),

                title=(
                    f"Balanced Network - {results['implementation']} | N={results['n_neurons']}, g={results['g']}, "
                    f"neurons/core={results['neurons_per_core']}\n"
                    f"p_conn={results['p_conn']}, input={results['rate_ext_input']} Hz | "
                    f"CV={results['cv']:.3f}, "
                    f"Exc={results['exc_firing_rate']:.2f} Hz, "
                    f"Inh={results['inh_firing_rate']:.2f} Hz \n"
                )
            )

    plt.savefig(
        f"balanced_network_N{results['n_neurons']}_g{results['g']}_core{results['neurons_per_core']}_"
        f"implementation{results['implementation']}_{results['timestamp']}.png"
    )

    plt.savefig(
        f"balanced_network_N{results['n_neurons']}_g{results['g']}_core{results['neurons_per_core']}_"
        f"implementation{results['implementation']}_{results['timestamp']}.pdf"
    )

def plot_results(results):

    for name, result in results.items():

        if result is None:
            continue

        plot_membrane_potential(result)

        plot_raster(result)


class TestSpiNNakerBalancedNetwork:
    """SpiNNaker code generation tests"""

    @pytest.fixture(autouse=True,
                    scope="module")
    
    def generate_code(self):
        codegen_opts = {"neuron_synapse_pairs": [{"neuron": "iaf_psc_exp_neuron",
                                                  "synapses": {"stdp_synapse": {"post_ports": ["post_spikes"]}}}],
                        "weight_variable": {"stdp_synapse": "w"}}

        files = [
            os.path.join("models", "neurons", "iaf_psc_exp_with_ignore_neuron.nestml"),
            os.path.join("models", "synapses", "stdp_additive_synapse.nestml")
        ]
        input_path = [os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.join(
            os.pardir, os.pardir, s))) for s in files]
        target_path = "spinnaker-target"
        install_path = "spinnaker-install"
        logging_level = "DEBUG"
        suffix = "_nestml"
        generate_spinnaker_target(input_path,
                                  target_path=target_path,
                                  install_path=install_path,
                                  logging_level=logging_level,
                                  suffix=suffix,
                                  codegen_opts=codegen_opts)



    def run_balanced_network(self, use_nestml_neuron, use_nestml_synapse):
        from python_models8.neuron.implementations.stdp_synapse_nestml_impl import stdp_synapse_nestmlDynamics as stdp_synapse_nestml
        from python_models8.neuron.builds.iaf_psc_exp_neuron_nestml import iaf_psc_exp_neuron_nestml

        t_sim = 1000    # total time to simulator for [ms]
        p_conn = .1    # connection probability
        rate_ext_input = 50.    # external input rate (eta parameter) [s⁻¹]
        n_neurons = 50
        n_exc = int(round(n_neurons * 0.8))
        n_inh = int(round(n_neurons * 0.2))
        g = 10.    # the ratio between excitation and inhibition
                # try -10 for asynchronous irregular activity. Try -1 for population-wide activity bursts
        neurons_per_core=16

        #Setup
        p.setup(timestep=1.0)

        p.set_number_of_neurons_per_core( p.SpikeSourcePoisson, 4)

        # Implementation name
        if use_nestml_neuron and use_nestml_synapse:
            implementation = "nestml_neuron_nestml_stdp"

        elif not use_nestml_neuron and use_nestml_synapse:
            implementation = "builtin_neuron_nestml_stdp"

        elif use_nestml_neuron and not use_nestml_synapse:
            implementation = "nestml_neuron_builtin_stdp"

        else:
            implementation = "builtin_neuron_builtin_stdp"


        if use_nestml_neuron:

            neuron_model = iaf_psc_exp_neuron_nestml()

            receptor_name_exc = "exc_spikes"
            receptor_name_inh = "inh_spikes"
            membranePot = "V_m"

            neuron_parameters = {
                "C_m": 250,
                "tau_m": 10,
                "tau_syn_inh": 2,
                "tau_syn_exc": 2,
                "refr_T": 2,
                "E_L": -70,
                "V_reset": -70,
                "V_th": -55
            }


        else:

            neuron_model = p.IF_curr_exp

            receptor_name_exc = "excitatory"
            receptor_name_inh = "inhibitory"
            membranePot = "v"

            neuron_parameters = {
                "tau_m": 10,
                "cm": 1E-3 * 250,
                "v_rest": -70,
                "v_reset": -70,
                "v_thresh": -55,
                "tau_syn_E": 2,
                "tau_syn_I": 2,
                "tau_refrac": 2
            }

        # Inıtial Weights
        weight_exc = 1E3 * 0.5
        weight_inh = -g * weight_exc
        weight_input = 1E3

        if not use_nestml_neuron:
            weight_exc *= 1E-3
            weight_inh *= 1E-3
            weight_input *= 1E-3



        #STDP Parameters
        learning_rate = 0.0

        tau_pre_trace = 20.0
        tau_post_trace = 20.0
        alpha = 1.0
        mu_plus = 1.0
        mu_minus = 1.0

        Wmin_exc = 0.0
        Wmax_exc = weight_exc * 2.0

        Wmin_inh = 0.0
        Wmax_inh = abs(weight_inh) * 2.0



        # p.set_number_of_neurons_per_core(neuron_model, neurons_per_core)

    
        # excitatory and inhibitory populations
        pop_exc = p.Population(n_exc, neuron_model, label="Excitatory", seed=1, additional_parameters={"max_atoms_per_core": neurons_per_core})
        pop_inh = p.Population(n_inh, neuron_model, label="Inhibitory", seed=2, additional_parameters={"max_atoms_per_core": neurons_per_core})
            
        pop_exc.set(**neuron_parameters)
        pop_inh.set(**neuron_parameters)


        # external stimulus to exc and inh populations
        stim_exc = p.Population(n_exc, p.SpikeSourcePoisson(rate=rate_ext_input), label="Stim_Exc", additional_parameters={"seed": 3})
        stim_inh = p.Population(n_inh, p.SpikeSourcePoisson(rate=rate_ext_input), label="Stim_Inh", additional_parameters={"seed": 4})

        p.Projection(stim_exc, pop_exc, p.OneToOneConnector(), p.StaticSynapse(weight=weight_input, delay=1.), receptor_type=receptor_name_exc)
        p.Projection(stim_inh, pop_inh, p.OneToOneConnector(), p.StaticSynapse(weight=weight_input, delay=1.), receptor_type=receptor_name_exc)


        # Exc and Inh Connections
        delays_exc = RandomDistribution("normal_clipped", mu=1.5, sigma=0.75, low=1.0, high=1.6)
        weights_exc = RandomDistribution("normal_clipped", mu=weight_exc, sigma=0.1, low=0.0, high=np.inf)
        conn_exc = p.FixedProbabilityConnector(p_conn)

        delays_inh = RandomDistribution("normal_clipped", mu=0.75, sigma=0.375, low=1.0, high=1.6)
        weights_inh = RandomDistribution("normal_clipped", mu=abs(weight_inh), sigma=0.1, low=0, high=np.inf)
        conn_inh = p.FixedProbabilityConnector(p_conn)

        # STDP synapses
        if use_nestml_synapse:
            stdp_ee = stdp_synapse_nestml(weight=weights_exc, delay=delays_exc)
            stdp_ei = stdp_synapse_nestml(weight=weights_exc, delay=delays_exc)
            stdp_ie = p.StaticSynapse(weight=weights_inh, delay=delays_inh)
            stdp_ii = p.StaticSynapse(weight=weights_inh, delay=delays_inh)

            print("STDP VARIABLES:")
            print(stdp_ee._nestml_model_variables)

            stdp_ee._nestml_model_variables["lambda"] = learning_rate

            # for stdp_model in [stdp_ee, stdp_ei, stdp_ie, stdp_ii]:

            #     stdp_model._nestml_model_variables["lambda"] = learning_rate
            #     stdp_model._nestml_model_variables["tau_tr_pre"] = tau_pre_trace
            #     stdp_model._nestml_model_variables["tau_tr_post"] = tau_post_trace
            #     stdp_model._nestml_model_variables["Wmin"] = Wmin_exc
            #     stdp_model._nestml_model_variables["Wmax"] = Wmax_exc
            print("STDP lambda:", stdp_ee._nestml_model_variables["lambda"])

            # Recurrent projections
            projection_ee=p.Projection(pop_exc, pop_exc, conn_exc, synapse_type=stdp_ee, receptor_type=receptor_name_exc)
            projection_ei=p.Projection(pop_exc, pop_inh, conn_exc, synapse_type=stdp_ei, receptor_type=receptor_name_exc)
            projection_ii=p.Projection(pop_inh, pop_inh, conn_inh, synapse_type=stdp_ii, receptor_type=receptor_name_inh)
            projection_ie=p.Projection(pop_inh, pop_exc, conn_inh, synapse_type=stdp_ie, receptor_type=receptor_name_inh)

        else: 
        # Built-in additive STDP

            timing_rule = p.SpikePairRule(
                tau_plus=tau_pre_trace,
                tau_minus=tau_post_trace,
                A_plus=learning_rate,
                A_minus=learning_rate
            )

            weight_rule_exc = p.AdditiveWeightDependence(
                w_min=Wmin_exc,
                w_max=Wmax_exc
            )

            weight_rule_inh = p.AdditiveWeightDependence(
                w_min=Wmin_inh,
                w_max=Wmax_inh
            )


            synapse_exc = p.STDPMechanism(
                timing_dependence=timing_rule,
                weight_dependence=weight_rule_exc,
                weight=weights_exc,
                delay=delays_exc
            )

            synapse_inh = p.StaticSynapse(weight=weights_inh, delay=delays_inh)


            projection_ee=p.Projection(pop_exc, pop_exc, conn_exc, synapse_type=synapse_exc, receptor_type=receptor_name_exc)
            projection_ei=p.Projection(pop_exc, pop_inh, conn_exc, synapse_type=synapse_exc, receptor_type=receptor_name_exc)
            projection_ii=p.Projection(pop_inh, pop_inh, conn_inh, synapse_type=synapse_inh, receptor_type=receptor_name_inh)
            projection_ie=p.Projection(pop_inh, pop_exc, conn_inh, synapse_type=synapse_inh, receptor_type=receptor_name_inh)



        # Initial membrane potentials
        if use_nestml_neuron:
            pop_exc.initialize(V_m=RandomDistribution("uniform", low=-65.0, high=-55.0))
            pop_inh.initialize(V_m=RandomDistribution("uniform", low=-65.0, high=-55.0))
        else:
            pop_exc.initialize(v=RandomDistribution("uniform", low=-65.0, high=-55.0))
            pop_inh.initialize(v=RandomDistribution("uniform", low=-65.0, high=-55.0))

        # Recording
        pop_exc[:5].record([membranePot])    # record only from the first 5 neurons
        pop_inh.record("spikes")
        pop_exc.record("spikes")

        start_time = time.perf_counter()
        p.run(t_sim)
        execution_time = time.perf_counter() - start_time

        pop_exc_spikes = pop_exc.get_data("spikes")
        pop_inh_spikes = pop_inh.get_data("spikes")
        v_neuron = pop_exc.get_data(membranePot)


        exc_spike_trains = (pop_exc_spikes.segments[0].spiketrains)
        inh_spike_trains = (pop_inh_spikes.segments[0].spiketrains)


        # Firing Rates
        exc_firing_rate = compute_average_firing_rate(exc_spike_trains, n_exc, t_sim)
        inh_firing_rate = compute_average_firing_rate(inh_spike_trains, n_inh, t_sim)

        cv = compute_cv_for_neurons(exc_spike_trains)

        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Finished {implementation} simulation.")

        w_ee = projection_ee.get("weight", format="float")
        w_ei = projection_ei.get("weight", format="float")
        w_ie = projection_ie.get("weight", format="float")
        w_ii = projection_ii.get("weight", format="float")

        mean_w_ee = np.mean(w_ee)
        mean_w_ei = np.mean(w_ei)
        mean_w_ie = np.mean(w_ie)
        mean_w_ii = np.mean(w_ii)

        p.end()

        return {
            "implementation": implementation,
            "timestamp": timestamp,

            "t_sim": t_sim,

            "n_neurons": n_neurons,
            "n_exc": n_exc,
            "n_inh": n_inh,

            "g": g,
            "p_conn": p_conn,
            "rate_ext_input": rate_ext_input,
            "neurons_per_core": neurons_per_core,

            "exc_firing_rate": exc_firing_rate,
            "inh_firing_rate": inh_firing_rate,
            "cv": cv,

            "execution_time": execution_time,

            "mean_w_ee": mean_w_ee,
            "mean_w_ei": mean_w_ei,
            "mean_w_ie": mean_w_ie,
            "mean_w_ii": mean_w_ii,

            "pop_exc_spikes": pop_exc_spikes,
            "pop_inh_spikes": pop_inh_spikes,

            "v_neuron": v_neuron,
            "membranePot": membranePot
        }


    def run_experiment(self):
        """
        Run reference and NESTML, compare their results,
        save results and generate plots.
        """

        # if use_builtin_stdp:
        #     reference = self.run_balanced_network(
        #         use_nestml_neuron=False,
        #         use_nestml_synapse=False
        #     )
        
        # else:
        #     reference = None


        builtin_neuron_nestml_stdp = self.run_balanced_network(
            use_nestml_neuron=False,
            use_nestml_synapse=False
        )


        # if use_builtin_stdp:
        #     nestml_neuron_builtin_stdp = self.run_balanced_network(
        #             use_nestml_neuron=True,
        #             use_nestml_synapse=False
        #         )
        # else:
        #     nestml_neuron_builtin_stdp = None


        # nestml = self.run_balanced_network(
        #     use_nestml_neuron=True,
        #     use_nestml_synapse=True
        # )

        results = {
            # "reference": reference,
            "builtin_neuron_nestml_stdp": builtin_neuron_nestml_stdp,
            # "nestml_neuron_builtin_stdp": nestml_neuron_builtin_stdp,
            # "nestml": nestml
        }

        if reference is not None:

            comparisons = compare_results(
                reference,
                results
            )
        else:
            comparisons = {}

        print_results(results, comparisons)

        save_results_to_csv(results)

        if reference is not None:

            save_comparisons_to_csv(
                reference,
                comparisons
            )

        plot_results(results)

        return results, comparisons

    # Pytest

    def test_spinnaker_balanced_network(self):

        results, comparisons = self.run_experiment()

        # Basic checks

        for name, result in results.items():

            if result is None:
                continue

            assert np.isfinite(result["exc_firing_rate"])
            assert np.isfinite(result["inh_firing_rate"])
            assert np.isfinite(result["cv"])

            assert result["exc_firing_rate"] > 0.0
            assert result["inh_firing_rate"] > 0.0

        # Reference vs other implementations
        if "reference" in comparisons:

            exc_rate_tolerance = 0.40
            inh_rate_tolerance = 0.40
            cv_tolerance = 0.20

            for name, comparison in comparisons.items():

                assert comparison["exc_relative_difference"] < exc_rate_tolerance
                assert comparison["inh_relative_difference"] < inh_rate_tolerance
                assert comparison["cv_absolute_difference"] < cv_tolerance