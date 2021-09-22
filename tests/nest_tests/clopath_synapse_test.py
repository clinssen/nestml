import nest
import numpy as np

from pynestml.frontend.pynestml_frontend import to_nest
from pynestml.utils.model_installer import install_nest


def run_simulation(neuron_model_name, synapse_model_name, module_name, _pre_spike_times, _post_spike_times, delay=1.0):
    nest.set_verbosity("M_ALL")
    nest.ResetKernel()
    try:
        print('Installing module: ', module_name)
        nest.Install(module_name)
    except BaseException:
        pass  # pass if the module is already loaded

    sim_time = max(np.amax(pre_spike_times), np.amax(post_spike_times)) + 5 * delay

    neurons = nest.Create(neuron_model_name, 2)
    pre_sg = nest.Create('spike_generator', params={'spike_times': _pre_spike_times})
    post_sg = nest.Create('spike_generator', params={'spike_times': _post_spike_times})

    spikes = nest.Create('spike_recorder')
    wr = nest.Create('weight_recorder')
    wr_ref = nest.Create('weight_recorder')
    nest.CopyModel(synapse_model_name, synapse_model_name + "_rec",
                   {"weight_recorder": wr[0], "w": 1., "the_delay": 1., "receptor_type": 0, "lambda": .001})

    multimeter_pre = nest.Create('multimeter')
    multimeter_post = nest.Create('multimeter',
                                  params={"record_from": "u_bar_plus__for_clopath_nestml"})

    nest.Connect(neurons[0], neurons[1], syn_spec={'synapse_model': synapse_model_name + "_rec"})
    nest.Connect(pre_sg, neurons[0], "one_to_one", syn_spec={"delay": 1.})
    nest.Connect(post_sg, neurons[1], "one_to_one", syn_spec={"delay": 1., "weight": 9999.})
    nest.Connect(multimeter_pre, neurons[0])
    nest.Connect(multimeter_post, neurons[1])
    nest.Connect(neurons, spikes)

    # Simulate
    nest.Simulate(sim_time)

    # Record u_bar_plus
    events_post = nest.GetStatus(multimeter_post, 'events')[0]
    times_post = events_post['times']
    u_bar_plus = events_post['u_bar_plus__for_clopath_nestml']

    print(u_bar_plus)


nest_install_path = nest.ll_api.sli_func("statusdict/prefix ::")
module_name = "nestml_clopath_synapse_module"
to_nest(input_path=["models/iaf_psc_delta.nestml", "models/clopath_synapse.nestml"],
        target_path="/tmp/nestml-clopath",
        logging_level="INFO",
        module_name=module_name,
        suffix="_nestml",
        codegen_opts={"neuron_parent_class": "StructuralPlasticityNode",
                      "neuron_parent_class_include": "structural_plasticity_node.h",
                      "neuron_synapse_pairs": [{"neuron": "iaf_psc_delta",
                                                "synapse": "clopath",
                                                "post_ports": ["post_spikes", ["v_clamp", "V_abs"]]}]})
install_nest("/tmp/nestml-clopath", nest_install_path)

neuron_model_name = "iaf_psc_delta_nestml__with_clopath_nestml"
synapse_model_name = "clopath_nestml__with_iaf_psc_delta_nestml"

post_spike_times = np.sort(np.unique(1 + np.round(10 * np.sort(np.abs(np.random.randn(10))))))
pre_spike_times = np.sort(np.unique(1 + np.round(10 * np.sort(np.abs(np.random.randn(10))))))

run_simulation(neuron_model_name, synapse_model_name, module_name, pre_spike_times, post_spike_times)
