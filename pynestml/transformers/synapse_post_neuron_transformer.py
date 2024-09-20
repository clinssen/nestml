# -*- coding: utf-8 -*-
#
# synapse_post_neuron_transformer.py
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

from __future__ import annotations

from typing import Any, Sequence, Mapping, Optional, Union

from pynestml.frontend.frontend_configuration import FrontendConfiguration
from pynestml.meta_model.ast_assignment import ASTAssignment
from pynestml.meta_model.ast_equations_block import ASTEquationsBlock
from pynestml.meta_model.ast_inline_expression import ASTInlineExpression
from pynestml.meta_model.ast_model import ASTModel
from pynestml.meta_model.ast_node import ASTNode
from pynestml.meta_model.ast_simple_expression import ASTSimpleExpression
from pynestml.meta_model.ast_variable import ASTVariable
from pynestml.symbols.symbol import SymbolKind
from pynestml.symbols.variable_symbol import BlockType
from pynestml.transformers.transformer import Transformer
from pynestml.utils.ast_utils import ASTUtils
from pynestml.utils.logger import Logger
from pynestml.utils.logger import LoggingLevel
from pynestml.utils.string_utils import removesuffix
from pynestml.visitors.ast_parent_visitor import ASTParentVisitor
from pynestml.visitors.ast_symbol_table_visitor import ASTSymbolTableVisitor
from pynestml.visitors.ast_higher_order_visitor import ASTHigherOrderVisitor
from pynestml.visitors.ast_visitor import ASTVisitor


class SynapsePostNeuronTransformer(Transformer):
    r"""In a (pre neuron, synapse, post neuron) tuple, process (synapse, post_neuron) to move all variables that are only triggered by postsynaptic events to the postsynaptic neuron."""

    _default_options = {
        "neuron_synapse_pairs": []
    }

    def __init__(self, options: Optional[Mapping[str, Any]] = None):
        super(Transformer, self).__init__(options)
        self._copy_custom_options(options)

    def _copy_custom_options(self, options):
        if options:
            if "delay_variable" in options:
                self._options["delay_variable"] = options["delay_variable"].copy()

            if "weight_variable" in options:
                self._options["weight_variable"] = options["weight_variable"].copy()

    def set_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        r"""Set options. "Eats off" any options that it knows how to set, and returns the rest as "unhandled" options."""
        unused_options = super().set_options(options)
        self._copy_custom_options(options)

        return unused_options

    def is_special_port(self, special_type: str, port_name: str, neuron_name: str, synapse_name: str) -> bool:
        """
        Check if a port by the given name is specified as connecting to the postsynaptic neuron. Only makes sense
        for synapses.
        """
        assert special_type in ["post", "vt"]
        if not "neuron_synapse_pairs" in self._options.keys():
            return False

        for neuron_synapse_pair in self._options["neuron_synapse_pairs"]:
            if not (neuron_name in [neuron_synapse_pair["neuron"], neuron_synapse_pair["neuron"] + FrontendConfiguration.suffix]
                    and synapse_name in [neuron_synapse_pair["synapse"], neuron_synapse_pair["synapse"] + FrontendConfiguration.suffix]):
                continue

            if not special_type + "_ports" in neuron_synapse_pair.keys():
                return False

            post_ports = neuron_synapse_pair[special_type + "_ports"]
            if not isinstance(post_ports, list):
                # only one port name given, not a list
                return port_name == post_ports

            for post_port in post_ports:
                if type(post_port) is not str and len(post_port) == 2:  # (syn_port_name, neuron_port_name) tuple
                    post_port = post_port[0]
                if type(post_port) is not str and len(post_port) == 1:  # (syn_port_name)
                    return post_port[0] == port_name
                if port_name == post_port:
                    return True

        return False

    def is_continuous_port(self, port_name: str, parent_node: ASTModel):
        for input_block in parent_node.get_input_blocks():
            for port in input_block.get_input_ports():
                if port.is_continuous() and port_name == port.get_name():
                    return True
        return False

    def is_post_port(self, port_name: str, neuron_name: str, synapse_name: str) -> bool:
        return self.is_special_port("post", port_name, neuron_name, synapse_name)

    def is_vt_port(self, port_name: str, neuron_name: str, synapse_name: str) -> bool:
        return self.is_special_port("vt", port_name, neuron_name, synapse_name)

    def get_spiking_post_port_names(self, synapse, neuron_name: str, synapse_name: str):
        post_port_names = []
        for input_block in synapse.get_input_blocks():
            for port in input_block.get_input_ports():
                if self.is_post_port(port.name, neuron_name, synapse_name) and port.is_spike():
                    post_port_names.append(port.get_name())
        return post_port_names

    def get_post_port_names(self, synapse, neuron_name: str, synapse_name: str):
        post_port_names = []
        for input_block in synapse.get_input_blocks():
            for port in input_block.get_input_ports():
                if self.is_post_port(port.name, neuron_name, synapse_name):
                    post_port_names.append(port.get_name())
        return post_port_names

    def get_vt_port_names(self, synapse, neuron_name: str, synapse_name: str):
        post_port_names = []
        for input_block in synapse.get_input_blocks():
            for port in input_block.get_input_ports():
                if self.is_vt_port(port.name, neuron_name, synapse_name):
                    post_port_names.append(port.get_name())
        return post_port_names

    def get_neuron_var_name_from_syn_port_name(self, port_name: str, neuron_name: str, synapse_name: str) -> Optional[str]:
        """
        Check if a port by the given name is specified as connecting to the postsynaptic neuron. Only makes sense for synapses.
        """
        if not "neuron_synapse_pairs" in self._options.keys():
            return False

        for neuron_synapse_pair in self._options["neuron_synapse_pairs"]:
            if not (neuron_name in [neuron_synapse_pair["neuron"], neuron_synapse_pair["neuron"] + FrontendConfiguration.suffix]
                    and synapse_name in [neuron_synapse_pair["synapse"], neuron_synapse_pair["synapse"] + FrontendConfiguration.suffix]):
                continue

            if not "post_ports" in neuron_synapse_pair.keys():
                return None

            post_ports = neuron_synapse_pair["post_ports"]

            for post_port in post_ports:
                if type(post_port) is not str and len(post_port) == 2:  # (syn_port_name, neuron_var_name) tuple
                    if port_name == post_port[0]:
                        return post_port[1]

            return None

        return None

    def get_all_variables_assigned_to(self, node):
        r"""Return a list of all variables that are assigned to in ``node``."""
        class ASTAssignedToVariablesFinderVisitor(ASTVisitor):
            _variable_names = []

            def __init__(self, synapse):
                super(ASTAssignedToVariablesFinderVisitor, self).__init__()
                self.synapse = synapse

            def visit_assignment(self, node):
                symbol = node.get_scope().resolve_to_symbol(node.get_variable().get_complete_name(), SymbolKind.VARIABLE)
                assert symbol is not None  # should have been checked in a CoCo before
                self._variable_names.append(node.get_variable().get_name())

        if node is None:
            return []

        visitor = ASTAssignedToVariablesFinderVisitor(node)
        node.accept(visitor)

        return visitor._variable_names

    def transform_neuron_synapse_pair_(self, neuron, synapse):
        r"""
        "Co-generation" or in-tandem generation of neuron and synapse code.

        Does not modify existing neurons or synapses, but returns lists with additional elements representing new pair neuron and synapse
        """
        new_neuron = neuron.clone()
        new_synapse = synapse.clone()

        new_neuron.parent_ = None    # set root element
        new_neuron.accept(ASTParentVisitor())
        new_synapse.parent_ = None    # set root element
        new_synapse.accept(ASTParentVisitor())
        new_neuron.accept(ASTSymbolTableVisitor())
        new_synapse.accept(ASTSymbolTableVisitor())

        assert len(new_neuron.get_equations_blocks()) <= 1, "Only one equations block per neuron supported for now."
        assert len(new_synapse.get_equations_blocks()) <= 1, "Only one equations block per synapse supported for now."
        assert len(new_neuron.get_state_blocks()) <= 1, "Only one state block supported per neuron for now."
        assert len(new_synapse.get_state_blocks()) <= 1, "Only one state block supported per synapse for now."
        assert len(new_neuron.get_update_blocks()) <= 1, "Only one update block supported per neuron for now."
        assert len(new_synapse.get_update_blocks()) <= 1, "Only one update block supported per synapse for now."

        #
        #   suffix for variables that will be transferred to neuron
        #

        var_name_suffix = "__for_" + synapse.get_name()

        #
        #   determine which variables and dynamics in synapse can be transferred to neuron
        #

        if synapse.get_state_blocks():
            all_state_vars = ASTUtils.all_variables_defined_in_block(synapse.get_state_blocks()[0])
        else:
            all_state_vars = []

        all_state_vars = [var.get_complete_name() for var in all_state_vars]

        # exclude certain variables from being moved:
        # exclude any variable assigned to in any block that is not connected to a postsynaptic port
        strictly_synaptic_vars = ["t"]      # "seed" this with the predefined variable t
        if self.option_exists("delay_variable") and removesuffix(synapse.get_name(), FrontendConfiguration.suffix) in self.get_option("delay_variable").keys() and self.get_option("delay_variable")[removesuffix(synapse.get_name(), FrontendConfiguration.suffix)]:
            strictly_synaptic_vars.append(self.get_option("delay_variable")[removesuffix(synapse.get_name(), FrontendConfiguration.suffix)])

        if self.option_exists("weight_variable") and removesuffix(synapse.get_name(), FrontendConfiguration.suffix) in self.get_option("weight_variable").keys() and self.get_option("weight_variable")[removesuffix(synapse.get_name(), FrontendConfiguration.suffix)]:
            strictly_synaptic_vars.append(self.get_option("weight_variable")[removesuffix(synapse.get_name(), FrontendConfiguration.suffix)])

        for input_block in new_synapse.get_input_blocks():
            for port in input_block.get_input_ports():
                if not self.is_post_port(port.name, neuron.name, synapse.name):
                    strictly_synaptic_vars += self.get_all_variables_assigned_to(synapse.get_on_receive_block(port.name))

        # exclude all variables that are assigned to in the ``update`` block
        for update_block in synapse.get_update_blocks():
            strictly_synaptic_vars += self.get_all_variables_assigned_to(update_block)

        # exclude all variables that depend on the ones that are not to be moved
        strictly_synaptic_vars_dependent = ASTUtils.recursive_dependent_variables_search(strictly_synaptic_vars, synapse)

        # do set subtraction
        syn_to_neuron_state_vars = list(set(all_state_vars) - (set(strictly_synaptic_vars) | set(strictly_synaptic_vars_dependent)))

        #
        #   collect all the variable/parameter/function/etc. names used in defining expressions of `syn_to_neuron_state_vars`
        #

        recursive_vars_used = ASTUtils.recursive_necessary_variables_search(syn_to_neuron_state_vars, synapse)
        new_neuron.recursive_vars_used = recursive_vars_used
        new_neuron._transferred_variables = [neuron_state_var + var_name_suffix
                                             for neuron_state_var in syn_to_neuron_state_vars]

        # all state variables that will be moved from synapse to neuron
        syn_to_neuron_state_vars = []
        for var_name in recursive_vars_used:
            if ASTUtils.get_state_variable_by_name(synapse, var_name) or ASTUtils.get_inline_expression_by_name(synapse, var_name):
                syn_to_neuron_state_vars.append(var_name)

        Logger.log_message(None, -1, "State variables that will be moved from synapse to neuron: " + str(syn_to_neuron_state_vars),
                           None, LoggingLevel.INFO)

        #
        #   collect all the parameters
        #

        if new_synapse.get_parameters_blocks():
            all_declared_params = [s.get_variables() for s in new_synapse.get_parameters_blocks()[0].get_declarations()]
        else:
            all_declared_params = []

        all_declared_params = sum(all_declared_params, [])
        all_declared_params = [var.name for var in all_declared_params]

        syn_to_neuron_params = [v for v in recursive_vars_used if v in all_declared_params]

        vars_used = []
        for var in syn_to_neuron_state_vars:
            # parameters used in the declarations of the state variables
            for state_block in synapse.get_state_blocks():
                decls = ASTUtils.get_declarations_from_block(var, state_block)
                for decl in decls:
                    if decl.has_expression():
                        vars_used.extend(ASTUtils.collect_variable_names_in_expression(decl.get_expression()))

            # parameters used in equations
            for equations_block in synapse.get_equations_blocks():
                vars_used.extend(ASTUtils.collects_vars_used_in_equation(var, equations_block))

        vars_used = list(set([str(var) for var in vars_used]))

        syn_to_neuron_params.extend([var for var in vars_used if var in all_declared_params])
        syn_to_neuron_params = list(set(syn_to_neuron_params))

        Logger.log_message(None, -1, "Parameters that will be copied from synapse to neuron: " + str(syn_to_neuron_params),
                           None, LoggingLevel.INFO)

        #
        #   collect all the internal parameters
        #

        # XXX: TODO

        #
        #   move defining equations for variables from synapse to neuron
        #

        if not new_synapse.get_equations_blocks():
            ASTUtils.create_equations_block(new_synapse)

        if not new_neuron.get_equations_blocks():
            ASTUtils.create_equations_block(new_neuron)

        post_port_names = []
        for input_block in new_synapse.get_input_blocks():
            for port in input_block.get_input_ports():
                if self.is_post_port(port.name, neuron.name, synapse.name):
                    post_port_names.append(port.name)

        for state_var in syn_to_neuron_state_vars:
            Logger.log_message(None, -1, "Moving state var defining equation(s) " + str(state_var),
                               None, LoggingLevel.INFO)
            # move the ODE so a solver will be generated for it by ODE-toolbox
            decls = ASTUtils.equations_from_block_to_block(state_var,
                                                           new_synapse.get_equations_blocks()[0],
                                                           new_neuron.get_equations_blocks()[0],
                                                           var_name_suffix,
                                                           mode="move")
            ASTUtils.add_suffix_to_variable_names2(post_port_names + syn_to_neuron_state_vars + syn_to_neuron_params, decls, var_name_suffix)
            ASTUtils.remove_state_var_from_integrate_odes_calls(new_synapse, state_var)
            # ASTUtils.add_integrate_odes_call_to_update_block(new_neuron, state_var)   # the moved state variables are never needed inside the neuron, their values are only read out from the side of the synapse. Therefore they do not have to be added to integrate_odes() calls; we just have to make sure the value has been updated before the end of the timestep
            # for now, moved variables are integrated separately in time in set_spiketime()

        #
        #    move initial values for equations
        #

        if syn_to_neuron_state_vars and not new_neuron.get_state_blocks():
            ASTUtils.create_state_block(new_neuron)

        for state_var in syn_to_neuron_state_vars:
            Logger.log_message(None, -1, "Moving state variables for equation(s) " + str(state_var),
                               None, LoggingLevel.INFO)
            ASTUtils.move_decls(var_name=state_var,
                                from_block=new_synapse.get_state_blocks()[0],
                                to_block=new_neuron.get_state_blocks()[0],
                                var_name_suffix=var_name_suffix,
                                block_type=BlockType.STATE,
                                mode="move")

        #
        #    move statements in post receive block from synapse to new_neuron
        #

        # XXX: TODO: do not use a new member variable (`extra_on_emit_spike_stmts_from_synapse`) for this, but add a new event handler block in the neuron

        # find all statements in post receive block
        collected_on_post_stmts = []

        for input_block in new_synapse.get_input_blocks():
            for port in input_block.get_input_ports():
                if self.is_post_port(port.name, new_neuron.name, new_synapse.name):
                    post_receive_blocks = ASTUtils.get_on_receive_blocks_by_input_port_name(new_synapse, port.name)
                    for post_receive_block in post_receive_blocks:
                        stmts = post_receive_block.get_block().get_stmts()
                        for stmt in stmts:
                            if stmt.is_small_stmt() \
                               and stmt.small_stmt.is_assignment() \
                               and ASTUtils.depends_only_on_vars(stmt.small_stmt.get_assignment().rhs, recursive_vars_used + all_declared_params) \
                               and stmt.small_stmt.get_assignment().get_variable().get_complete_name() in syn_to_neuron_params + syn_to_neuron_state_vars:
                                Logger.log_message(None, -1, "\tMoving statement " + str(stmt).strip(), None, LoggingLevel.INFO)

                                collected_on_post_stmts.append(stmt)

                                stmt.scope = new_neuron.scope
                                stmt.small_stmt.scope = new_neuron.scope
                                stmt.small_stmt.get_assignment().scope = new_neuron.scope
                                stmt.small_stmt.get_assignment().get_variable().scope = new_neuron.scope

                        for stmt in collected_on_post_stmts:
                            stmts.pop(stmts.index(stmt))

        new_neuron.extra_on_emit_spike_stmts_from_synapse = collected_on_post_stmts

        # XXX: TODO: add parameters used in stmts to parameters to be copied

        vars_used = list(set([str(v) for v in vars_used]))
        syn_to_neuron_params.extend([v for v in vars_used if v in [p + var_name_suffix for p in all_declared_params]])
        syn_to_neuron_params = list(set(syn_to_neuron_params))

        #
        #   replace ``continuous`` type input ports that are connected to postsynaptic neuron with suffixed external variable references
        #

        Logger.log_message(
            None, -1, "In synapse: replacing ``continuous`` type input ports that are connected to postsynaptic neuron with external variable references", None, LoggingLevel.INFO)
        post_connected_continuous_input_ports = []
        post_variable_names = []
        for input_block in synapse.get_input_blocks():
            for port in input_block.get_input_ports():
                if self.is_post_port(port.get_name(), neuron.name, synapse.name) and self.is_continuous_port(port.get_name(), synapse):
                    post_connected_continuous_input_ports.append(port.get_name())
                    post_variable_names.append(self.get_neuron_var_name_from_syn_port_name(
                        port.get_name(), neuron.name, synapse.name))

        for state_var, alternate_name in zip(post_connected_continuous_input_ports, post_variable_names):
            Logger.log_message(None, -1, "\t• Replacing variable " + str(state_var), None, LoggingLevel.INFO)
            ASTUtils.replace_with_external_variable(state_var, new_synapse, "", new_synapse.get_equations_blocks()[0].get_scope(), alternate_name)

        #
        #    copy parameters
        #

        if not new_neuron.get_parameters_blocks():
            ASTUtils.create_parameters_block(new_neuron)

        Logger.log_message(None, -1, "Copying parameters from synapse to neuron...", None, LoggingLevel.INFO)
        for param_var in syn_to_neuron_params:
            decls = ASTUtils.move_decls(param_var,
                                        new_synapse.get_parameters_blocks()[0],
                                        new_neuron.get_parameters_blocks()[0],
                                        var_name_suffix=var_name_suffix,
                                        block_type=BlockType.PARAMETERS,
                                        mode="copy")

        #
        #   add suffix to variables in moved update statements
        #

        Logger.log_message(
            None, -1, "Adding suffix to variables in spike updates", None, LoggingLevel.INFO)

        for stmt in new_neuron.extra_on_emit_spike_stmts_from_synapse:
            if stmt.small_stmt.is_assignment():
                new_name: str = stmt.small_stmt.get_assignment().get_variable().get_name() + var_name_suffix
                stmt.small_stmt.get_assignment().get_variable().set_name(new_name)

        #
        #    replace occurrences of the variables in expressions in the original synapse with calls to the corresponding neuron getters
        #

        # make sure the moved symbols can be resolved in the scope of the neuron (that's where ``ASTExternalVariable._altscope`` will be pointing to)
        ast_symbol_table_visitor = ASTSymbolTableVisitor()
        ast_symbol_table_visitor.after_ast_rewrite_ = True
        new_neuron.accept(ast_symbol_table_visitor)

        Logger.log_message(
            None, -1, "In synapse: replacing variables with suffixed external variable references", None, LoggingLevel.INFO)
        for state_var in syn_to_neuron_state_vars:
            Logger.log_message(None, -1, "\t• Replacing variable " + str(state_var), None, LoggingLevel.INFO)
            ASTUtils.replace_with_external_variable(state_var, new_synapse, var_name_suffix, new_neuron.get_equations_blocks()[0].get_scope())

        #
        #     rename neuron
        #

        name_separator_str = "__with_"

        new_neuron_name = neuron.get_name() + name_separator_str + synapse.get_name()
        new_neuron.set_name(new_neuron_name)

        #
        #    rename synapse
        #

        new_synapse_name = synapse.get_name() + name_separator_str + neuron.get_name()
        new_synapse.set_name(new_synapse_name)
        new_synapse.paired_neuron = new_neuron
        new_neuron.paired_synapse = new_synapse
        new_neuron.paired_synapse_original_model = synapse

        base_neuron_name = removesuffix(neuron.get_name(), FrontendConfiguration.suffix)
        base_synapse_name = removesuffix(synapse.get_name(), FrontendConfiguration.suffix)

        new_synapse.post_port_names = self.get_post_port_names(synapse, base_neuron_name, base_synapse_name)
        new_synapse.spiking_post_port_names = self.get_spiking_post_port_names(synapse, base_neuron_name, base_synapse_name)
        new_synapse.vt_port_names = self.get_vt_port_names(synapse, base_neuron_name, base_synapse_name)

        #
        #    add modified versions of neuron and synapse to list
        #

        new_neuron.accept(ASTParentVisitor())
        new_synapse.accept(ASTParentVisitor())
        ast_symbol_table_visitor = ASTSymbolTableVisitor()
        ast_symbol_table_visitor.after_ast_rewrite_ = True
        new_neuron.accept(ast_symbol_table_visitor)
        new_synapse.accept(ast_symbol_table_visitor)

        ASTUtils.update_blocktype_for_common_parameters(new_synapse)

        Logger.log_message(None, -1, "Successfully constructed neuron-synapse pair "
                           + new_neuron.name + ", " + new_synapse.name, None, LoggingLevel.INFO)

        return new_neuron, new_synapse

    def transform(self, models: Union[ASTNode, Sequence[ASTNode]]) -> Union[ASTNode, Sequence[ASTNode]]:
        # check that there are no convolutions or kernels in the model (these should have been transformed out by the ConvolutionsTransformer)
        for model in models:
            for equations_block in model.get_equations_blocks():
                assert len(equations_block.get_kernels()) == 0, "Kernels and convolutions should have been removed by ConvolutionsTransformer"

        # transform each (neuron, synapse) pair
        for neuron_synapse_pair in self.get_option("neuron_synapse_pairs"):
            neuron_name = neuron_synapse_pair["neuron"]
            synapse_name = neuron_synapse_pair["synapse"]
            neuron = ASTUtils.find_model_by_name(neuron_name + FrontendConfiguration.suffix, models)
            if neuron is None:
                raise Exception("Neuron used in pair (\"" + neuron_name + "\") not found")  # XXX: log error

            synapse = ASTUtils.find_model_by_name(synapse_name + FrontendConfiguration.suffix, models)
            if synapse is None:
                raise Exception("Synapse used in pair (\"" + synapse_name + "\") not found")  # XXX: log error

            new_neuron, new_synapse = self.transform_neuron_synapse_pair_(neuron, synapse)
            models.append(new_neuron)
            models.append(new_synapse)

        # remove the synapses used in neuron-synapse pairs, as they can potentially not be generated independently of a neuron and would otherwise result in an error
        for neuron_synapse_pair in self.get_option("neuron_synapse_pairs"):
            synapse_name = neuron_synapse_pair["synapse"]
            synapse = ASTUtils.find_model_by_name(synapse_name + FrontendConfiguration.suffix, models)
            if synapse:
                model_idx = models.index(synapse)
                models.pop(model_idx)

        return models
