Application notes
=================

This page contains several sections containing various application notes: tips and recommendations for implementing neuron and synapse models in NESTML.

For example models written in NESTML, see the :ref:`Models library`.

For examples of how to use NESTML and NESTML models, see the :ref:`Tutorials` as well as the `https://github.com/nest/nestml/tree/main/tests <NESTML unit tests>`_.


Implementing refractoriness in neuron models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to model an absolute refractory state, in which the neuron cannot fire action potentials, different approaches can be used. In general, an extra parameter (say, ``refr_T``) is introduced, that defines the duration of the refractory period. A new state variable (say, ``refr_t``) can then act as a timer, counting the time of the refractory period that has already elapsed. The dynamics of ``refr_t`` could be specified in the ``update`` block, as follows:

.. code-block:: nestml

   update:
       refr_t -= resolution()

The test for refractoriness can then be added in the ``onCondition`` block as follows:

.. code-block:: nestml

   # if not refractory and threshold is crossed...
   onCondition(refr_t <= 0 ms and V_m > V_th):
       V_m = E_L    # Reset the membrane potential
       refr_t = refr_T    # Start the refractoriness timer
       emit_spike()

The disadvantage of this method is that it requires a call to the ``resolution()`` function, which is only supported by fixed-timestep simulators, and furthermore the timer is always counting, even when the neuron is not refractory anymore. To write the model in a more generic way, the refractoriness timer can alternatively be expressed as an ODE, which represents the timer in continuous-time, counting down to zero at a rate of one (milli)second per (milli)second:

.. code-block:: nestml

   equations:
       refr_t' = -1

During the refractory period, the membrane potential should typically remain clamped to the reset or leak potential. It depends on the intended behavior of the model whether the synaptic currents and conductances also continue to be integrated or whether they are reset, and whether incoming spikes during the refractory period are taken into account or ignored.

In order to hold the membrane potential at the reset voltage during refractoriness, it can be simply excluded from the integration call:

.. code-block:: nestml

   equations:
       I_syn' = ...
       V_m' = ...
       refr_t' = -1

   update:
       if refr_t > 0 ms:
           # neuron is absolute refractory, do not evolve V_m
           integrate_odes(I_syn, refr_t)
       else:
           # neuron not refractory
           integrate_odes(I_syn, V_m)

Note that in some cases, the finite resolution by which real numbers are expressed (as floating point numbers) in computers, can cause unexpected behaviors. If the simulation resolution is not exactly representable as a float (say, :math:`\Delta t` = 0.1 ms) then it could be the case that after 20 simulation steps, the timer has not reached zero, but a very small value very close to zero (say, 0.00000001 ms), causing the refractory period to end only in the next timestep. If this kind of behavior is undesired, the simulation resolution and refractory period can be chosen as powers of two (which can be represented exactly as floating points), or a small "epsilon" value can be included in the comparison in the model:

.. code-block:: nestml

   parameters:
       float_epsilon ms = 1E-9 ms

   onCondition(refr_t <= float_epsilon ...):
       # ...

       
Modeling synapses in NESTML
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conceptually, a synapse model formalises the interaction between two (or more) neurons. In biophysical terms, they may contain some elements that are part of the postsynaptic neuron (such as the postsynaptic density) as well as the presynaptic neuron (such as the vesicle pool), or external factors such as the concentration of an extracellular diffusing factor. We will discuss in detail the spike-timing dependent plasticity (STDP) model and some of its variants.

From the modeling point of view, a synapse shares many of the same behaviours of a neuron: it has parameters and internal state variables, can communicate over input and output ports, and its dynamics and responses can be described by differential equations, kernels and as an algorithm. Typically, there is a single spiking input port and a single spiking output port.

.. Attention:: The NEST Simulator platform target has some additional constraints, such as precluding updates on a regular time grid. See :ref:`The NEST target` for more details.

Key to writing the synapse model is the requirement that the event handler for the spiking input port is responsible for submitting the event to the (spiking) output port.

Note that the synaptic strength ("weight") variable is of type real; if the type were given in more specific units, such as nS or pA, the synapse model would only be compatible with either a conductance or current-based postsynaptic neuron model.

Input and output ports
######################

Depending on whether the plasticity rule depends only on pre-, or on both pre- and postsynaptic activity, one or two input ports are defined. Synapses always have only one (spiking) output port.

.. code-block:: nestml

   input:
       pre_spikes <- spike
       post_spikes <- spike

   output:
       spike


Presynaptic spike event handler
###############################

Typically, it is the responsibility of the event handler for the spiking input port to create an event at the (spiking) output port. This can be done using the predefined ``emit_spike()`` function, which takes an (optional) parameter that for synapses typically corresponds to the weight ``w``.

The corresponding event handler has the general structure:

.. code-block:: nestml

   state:
       w real = 1

   onReceive(pre_spikes):
       print("Info: processing a presynaptic spike at time t = {t}")
       # ... plasticity dynamics go here ...
       emit_spike(w)

The statements in the event handler will be executed when the event occurs. If synaptic plasticity modifies the weight of the synapse, the weight update could (but does not have to) take place before calling ``emit_spike()`` with the updated weight.

State variables (in particular, synaptic "trace" variables as often used in plasticity models) can be updated in the event handler as follows:

.. code-block:: nestml

   state:
       tr_pre real = 0

   onReceive(post_spikes):
       print("Info: processing a postsynaptic spike at time t = {t}")
       tr_pre += 1

   equations:
       tr_pre' = -tr_pre / tau_tr

Equivalently, the trace can be defined as a convolution between a trace kernel and the spiking input port:

.. code-block:: nestml

   equations:
       kernel tr_pre_kernel = exp(-t / tau_tr)
       inline tr_pre real = convolve(tr_pre_kernel, pre_spikes)


Postsynaptic spike event handler
################################

Some plasticity rules are defined in terms of postsynaptic spike activity. A corresponding additional spiking input port and event handler (and convolutions) can be defined in the NESTML model:

.. code-block:: nestml

   input:
       pre_spikes <- spike  # (same as before)
       post_spikes <- spike

   onReceive(post_spikes):
       print("Info: processing a postsynaptic spike at time t = {t}")
       # ... plasticity dynamics go here ...


Sharing parameters between synapses
###################################

If one or more synapse parameters are the same across a population (homogeneous), then sharing the parameter value between all synapses can save vast amounts of memory. To mark a particular parameter as homogeneous, use the `@homogeneous` decorator keyword. This can be done on a per-parameter basis.

By default, parameters are heterogeneous which means can be set on a per-synapse basis by the user.

For example:

.. code-block:: nestml

   parameters:
       a real = pi        @homogeneous
       b integer = 42     @heterogeneous  # the default!


Third-factor plasticity
#######################

The postsynaptic trace value in the models so far is assumed to correspond to a property of the postsynaptic neuron, but it is specified in the synapse model. Some synaptic plasticity rules require access to a postsynaptic value that cannot be specified as part of the synapse model, but is a part of the (postsynaptic) neuron model.

An example would be a neuron that generates dendritic action potentials. (For more details about this neuron model, please see the tutorial https://nestml.readthedocs.io/en/latest/tutorials/active_dendrite/nestml_active_dendrite_tutorial.html.) The synapse could need access to the postsynaptic dendritic current.

To make this "third factor" value available in the synapse model, begin by defining an appropriate input port:

.. code-block:: nestml

   input:
       I_post_dend pA <- continuous

In the synapse, the value will be referred to as ``I_post_dend`` and can be used in equations and expressions. In this example, we will use it as a simple gating variable between 0 and 1, that can disable or enable weight updates in a graded manner:

.. code-block:: nestml

   onReceive(post_spikes):
       w_ real = # [...] normal STDP update rule
       w_ = (I_post_dend / I_post_dend_peak) * w_
            + (1 - I_post_dend / I_post_dend_peak) * w    # "gating" of the weight update

NESTML needs to be invoked so that it generates code for neuron and synapse together. Additionally, specify the ``"post_ports"`` entry to connect the input port on the synapse with the right variable of the neuron (see :ref:`Generating code`). Passing this as a code generator option facilitates combining models from different sources, where the naming conventions can be different between the neuron and synapse model.

In this example, the ``I_dend`` state variable of the neuron will be simply an exponentially decaying function of time, which can be clamped at predefined times in the simulation script. By inspecting the magnitude of the weight updates, we see that the synaptic plasticity is indeed being gated by the neuronal state variable ("third factor") ``I_dend``.

.. figure:: https://raw.githubusercontent.com/nest/nestml/main/doc/fig/stdp_triplet_synapse_test.png

For a full example, please see :doc:`Third-factor modulated STDP </tutorials/stdp_third_factor_active_dendrite/stdp_third_factor_active_dendrite>`.


Generating code
###############

Co-generation of neuron and synapse
-----------------------------------

Most plasticity models, including all of the STDP variants discussed above, depend on the storage and maintenance of "trace" values, that record the history of pre- and postsynaptic spiking activity. The trace dynamics and parameters are part of the synaptic plasticity rule that is being modeled, so logically belong in the NESTML synapse model. However, if each synapse maintains pre- and post traces for its connected partners, and considering that a single neuron may have on the order of thousands of synapses connected to it, these traces would be stored and computed redundantly. Instead of keeping them as part of the synaptic state during simulation, they more logically belong to the neuronal state.

To prevent this redundancy, a fully automated dependency analysis is run during code generation, that identifies those variables that depend exclusively on postsynaptic spikes, and moves them into the postsynaptic neuron model. For this to work, the postsynaptic neuron model used needs to be known at the time of synaptic code generation. Thus, we need to generate code "in tandem" now for connected neuron and synapse models, hence the name "co-generation".

.. figure:: https://raw.githubusercontent.com/nest/nestml/d4bf4f521d726dd638e8a264c7253a5746bcaaae/doc/fig/neuron_synapse_co_generation.png

   (a) Without co-generation: neuron and synapse models are treated independently. (b) co-generation: the code generator knows which neuron types will be connected using which synapse types, and treats these as pairs rather than independently.

To indicate which neurons will be connected to by which synapses during simulation, a list of such (neuron, synapse) pairs is passed to the code generator. This list is encoded as a JSON file. For example, if we want to use the "stdp" synapse model, connected to an "iaf_psc_exp" neuron, we would write the following:

.. code-block:: json

   {
     "neuron_synapse_pairs": [["iaf_psc_exp_neuron", "stdp_synapse"]]
   }

This file can then be passed to NESTML when generating code on the command line. If the JSON file is named ``nest_code_generator_opts_triplet.json``:

.. code:: sh

   nestml --input_path my_models/ --codegen_opts=nest_code_generator_opts_triplet.json

Further integration with NEST Simulator is planned, to achieve a just-in-time compilation/build workflow. This would automatically generate a list of these pairs and automatically generate the requisite JSON file.


.. figure:: https://raw.githubusercontent.com/nest/nestml/main/doc/fig/code_gen_opts.png
   :scale: 50 %
   :align: center

   Code generator options instruct the target platform code generator (in this case, NEST) how to process the models.



References
----------

.. [1] Morrison A., Diesmann M., and Gerstner W. (2008) Phenomenological
       models of synaptic plasticity based on spike timing,
       Biol. Cybern. 98, 459--478

.. [2] Front. Comput. Neurosci., 23 November 2010 | https://doi.org/10.3389/fncom.2010.00141 Enabling functional neural circuit simulations with distributed computing of neuromodulated plasticity, Wiebke Potjans, Abigail Morrison and Markus Diesmann

.. [3] Rubin, Lee and Sompolinsky. Equilibrium Properties of Temporally Asymmetric Hebbian Plasticity. Physical Review Letters, 8 Jan 2001, Vol 86, No 2

.. [4] Pfister JP, Gerstner W (2006). Triplets of spikes in a model of spike timing-dependent plasticity.  The Journal of Neuroscience 26(38):9673-9682. DOI: https://doi.org/10.1523/JNEUROSCI.1425-06.2006

.. [5] Potjans W, Morrison A and Diesmann M (2010) Enabling functional neural circuit simulations with distributed computing of neuromodulated plasticity. Front. Comput. Neurosci. 4:141. doi: 10.3389/fncom.2010.00141
