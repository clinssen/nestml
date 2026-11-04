# -*- coding: utf-8 -*-
#
# test_unit_system.py
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

import os
import pytest

from astropy import units as u

from pynestml.codegeneration.printers.nestml_printer import NESTMLPrinter
from pynestml.frontend.pynestml_frontend import generate_target
from pynestml.symbol_table.symbol_table import SymbolTable
from pynestml.symbols.predefined_functions import PredefinedFunctions
from pynestml.symbols.predefined_types import PredefinedTypes
from pynestml.symbols.predefined_units import PredefinedUnits
from pynestml.symbols.predefined_variables import PredefinedVariables
from pynestml.symbols.real_type_symbol import RealTypeSymbol
from pynestml.symbols.symbol import SymbolKind
from pynestml.symbols.unit_type_symbol import UnitTypeSymbol
from pynestml.transformers.non_dimensionalisation_transformer import NonDimensionalisationTransformer, get_default_unit_system_path
from pynestml.utils.ast_source_location import ASTSourceLocation
from pynestml.utils.logger import Logger, LoggingLevel
from pynestml.utils.model_parser import ModelParser
from pynestml.utils.unit_system import UnitSystem, UnitSystemException


NEST_UNIT_SYSTEM = get_default_unit_system_path("NEST")
SI_UNIT_SYSTEM = get_default_unit_system_path("PYTHON_STANDALONE")


class TestUnitSystem:
    r"""
    Test physical units and quantities, and their removal from models by the NonDimensionalisationTransformer.
    """

    @pytest.fixture(scope="function", autouse=True)
    def setUp(self, request):
        SymbolTable.initialize_symbol_table(ASTSourceLocation(start_line=0, start_column=0, end_line=0, end_column=0))

        PredefinedUnits.register_units()
        PredefinedTypes.register_types()
        PredefinedVariables.register_variables()
        PredefinedFunctions.register_functions()

        Logger.init_logger(LoggingLevel.INFO)

        request.cls.printer = NESTMLPrinter()

    #
    #   helper functions
    #

    def parse_resource(self, fname: str):
        model = ModelParser.parse_file(os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), "resources")), fname))
        assert model is not None
        return model.get_model_list()[0]

    def parse_string(self, tmp_path, model_str: str):
        fname = os.path.join(tmp_path, "model.nestml")
        with open(fname, "w") as f:
            f.write(model_str)

        model = ModelParser.parse_file(fname)
        assert model is not None
        assert len(Logger.get_messages(model.get_model_list()[0], LoggingLevel.ERROR)) == 0
        return model.get_model_list()[0]

    def transform(self, model, unit_system=NEST_UNIT_SYSTEM):
        metadata = {}
        models = NonDimensionalisationTransformer({"unit_system": unit_system}).transform([model], metadata)
        assert len(models) == 1
        return models[0], metadata

    def get_first_statement_in_update_block(self, model):
        return model.get_update_blocks()[0].get_stmts_body().get_stmts()[0]

    def print_rhs_of_first_assignment_in_update_block(self, model):
        assignment = self.get_first_statement_in_update_block(model).small_stmt.get_assignment()
        return self.printer.print(assignment.get_expression())

    def print_rhs_of_first_declaration_in_state_block(self, model):
        declaration = model.get_state_blocks()[0].get_declarations()[0]
        return self.printer.print(declaration.get_expression())

    def print_rhs_of_declaration(self, model, var_name: str):
        for block in model.get_state_blocks() + model.get_parameters_blocks() + model.get_internals_blocks():
            for decl in block.get_declarations():
                for var in decl.get_variables():
                    if var.get_complete_name() == var_name:
                        return self.printer.print(decl.get_expression())

        raise Exception("Declaration for variable \"" + var_name + "\" not found")

    def print_rhs_of_ode(self, model, var_name: str):
        for ode in model.get_equations_blocks()[0].get_ode_equations():
            if ode.get_lhs().get_complete_name() == var_name:
                return self.printer.print(ode.get_rhs())

        raise Exception("ODE for variable \"" + var_name + "\" not found")

    #
    #   conversion factors
    #

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "10000.0"), (SI_UNIT_SYSTEM, "10.0")])
    def test_expression_after_magnitude_conversion_in_direct_assignment(self, unit_system, expected):
        model, _ = self.transform(self.parse_resource("DirectAssignmentWithDifferentButCompatibleUnits.nestml"), unit_system)
        assert self.print_rhs_of_first_assignment_in_update_block(model) == expected

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "10000.0 + 5.0 + 20000.0 + 1000000.0"), (SI_UNIT_SYSTEM, "10.0 + 0.005 + 20.0 + 1000.0")])
    def test_expression_after_nested_magnitude_conversion_in_direct_assignment(self, unit_system, expected):
        model, _ = self.transform(self.parse_resource("DirectAssignmentWithDifferentButCompatibleNestedUnits.nestml"), unit_system)
        assert self.print_rhs_of_first_assignment_in_update_block(model) == expected

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "1200.0"), (SI_UNIT_SYSTEM, "1.2")])
    def test_expression_after_magnitude_conversion_in_compound_assignment(self, unit_system, expected):
        model, _ = self.transform(self.parse_resource("CompoundAssignmentWithDifferentButCompatibleUnits.nestml"), unit_system)
        assert self.print_rhs_of_first_assignment_in_update_block(model) == expected

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "10000.0"), (SI_UNIT_SYSTEM, "10.0")])
    def test_expression_after_magnitude_conversion_in_declaration(self, unit_system, expected):
        model, _ = self.transform(self.parse_resource("DeclarationWithDifferentButCompatibleUnitMagnitude.nestml"), unit_system)
        assert self.print_rhs_of_first_declaration_in_state_block(model) == expected

    def test_expression_after_type_conversion_in_declaration(self):
        model = self.parse_resource("DeclarationWithDifferentButCompatibleUnits.nestml")
        declaration = model.get_state_blocks()[0].get_declarations()[0]
        assert declaration.get_expression().type.unit.unit == u.mV

        # (10 pA) / (100 pF) * (1 ms) = 0.1 mV
        model, _ = self.transform(model, NEST_UNIT_SYSTEM)
        assert self.print_rhs_of_first_declaration_in_state_block(model) == "(10.0) / (100.0) * (1.0)"

        model, _ = self.transform(self.parse_resource("DeclarationWithDifferentButCompatibleUnits.nestml"), SI_UNIT_SYSTEM)
        assert self.print_rhs_of_first_declaration_in_state_block(model) == "0.001 * ((10.0) / (100.0) * (1.0))"

    def test_declaration_with_same_variable_name_as_unit(self):
        Logger.init_logger(LoggingLevel.DEBUG)

        generate_target(input_path=os.path.join(os.path.realpath(os.path.join(os.path.dirname(__file__), "resources")), "DeclarationWithSameVariableNameAsUnit.nestml"), target_platform="NONE", logging_level="DEBUG")

        assert len(Logger.get_messages("BlockTest", LoggingLevel.ERROR)) == 0
        assert len(Logger.get_messages("BlockTest", LoggingLevel.WARNING)) == 3

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "foo(10000.0)"), (SI_UNIT_SYSTEM, "foo(10.0)")])
    def test_expression_after_magnitude_conversion_in_standalone_function_call(self, unit_system, expected):
        model, _ = self.transform(self.parse_resource("FunctionCallWithDifferentButCompatibleUnits.nestml"), unit_system)
        function_call = self.get_first_statement_in_update_block(model).small_stmt.get_function_call()
        assert self.printer.print(function_call) == expected

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "foo(10000.0)"), (SI_UNIT_SYSTEM, "foo(10.0)")])
    def test_expression_after_magnitude_conversion_in_rhs_function_call(self, unit_system, expected):
        model, _ = self.transform(self.parse_resource("RhsFunctionCallWithDifferentButCompatibleUnits.nestml"), unit_system)
        assert self.print_rhs_of_first_assignment_in_update_block(model) == expected

    @pytest.mark.parametrize("unit_system, expected_unit", [(NEST_UNIT_SYSTEM, u.mV), (SI_UNIT_SYSTEM, u.V)])
    def test_return_stmt_after_magnitude_conversion_in_function_body(self, unit_system, expected_unit):
        r"""The function parameter is declared in mV and the return type in V; both are expressed in the same unit after the transformation, so no conversion is necessary."""
        model, _ = self.transform(self.parse_resource("FunctionBodyReturnStatementWithDifferentButCompatibleUnits.nestml"), unit_system)
        func = model.get_functions()[0]
        return_expression = func.get_stmts_body().get_stmts()[0].small_stmt.get_return_stmt().get_expression()
        assert self.printer.print(return_expression) == "bar"
        assert func.get_return_type().is_real
        assert func.get_return_type().get_physical_unit() == expected_unit
        assert func.get_parameters()[0].get_data_type().get_physical_unit() == expected_unit

    #
    #   physical quantities
    #

    def test_quantity_declarations(self, tmp_path):
        model = self.parse_string(tmp_path, """
model quantity_test_neuron:
    state:
        V_m "electrical potential" = -70 mV
        refr_t time = 2 ms

    parameters:
        C_m "electrical capacitance" = 250 pF
        rate frequency = 10 Hz
        k "electrical potential" / time = 1 mV/ms
        g "electrical conductance" = 1 uS
        x real = 42.
""")

        expected_si_units = {"V_m": u.V, "refr_t": u.s, "C_m": u.F, "rate": u.Hz, "k": u.V / u.s, "g": u.S}
        for var_name, expected_si_unit in expected_si_units.items():
            symbol = model.get_scope().resolve_to_symbol(var_name, SymbolKind.VARIABLE)
            assert isinstance(symbol.get_type_symbol(), UnitTypeSymbol)
            assert symbol.get_type_symbol().astropy_unit == expected_si_unit

        assert isinstance(model.get_scope().resolve_to_symbol("x", SymbolKind.VARIABLE).get_type_symbol(), RealTypeSymbol)

        # the NESTML printer reproduces the quantities
        model_str = self.printer.print(model)
        assert "V_m \"electrical potential\" = -70 mV" in model_str
        assert "refr_t time = 2 ms" in model_str
        assert "k \"electrical potential\"/time = 1 mV/ms" in model_str

        # the printed model can be parsed again
        model2 = self.parse_string(tmp_path, model_str)
        assert model2.equals(model)

        # transform with NEST units
        model, metadata = self.transform(model, NEST_UNIT_SYSTEM)
        expected = {"V_m": ("-70.0", u.mV), "refr_t": ("2.0", u.ms), "C_m": ("250.0", u.pF), "rate": ("10.0", u.Hz), "k": ("1.0", u.mV / u.ms), "g": ("1000.0", u.nS), "x": ("42.0", None)}
        for var_name, (expected_value, expected_unit) in expected.items():
            assert self.print_rhs_of_declaration(model, var_name) == expected_value
            symbol = model.get_scope().resolve_to_symbol(var_name, SymbolKind.VARIABLE)
            assert isinstance(symbol.get_type_symbol(), RealTypeSymbol)
            assert symbol.get_physical_unit() == expected_unit

        # the original model is retained in the metadata
        original_model = metadata["quantity_test_neuron"]["original_model"]
        assert isinstance(original_model.get_scope().resolve_to_symbol("V_m", SymbolKind.VARIABLE).get_type_symbol(), UnitTypeSymbol)
        assert "V_m \"electrical potential\" = -70 mV" in self.printer.print(original_model)

    def test_quantity_declarations_si(self, tmp_path):
        model = self.parse_string(tmp_path, """
model quantity_test_neuron:
    state:
        V_m "electrical potential" = -70 mV

    parameters:
        C_m "electrical capacitance" = 250 pF
        g "electrical conductance" = 1 uS
""")
        model, _ = self.transform(model, SI_UNIT_SYSTEM)
        expected = {"V_m": ("-0.07", u.V), "C_m": ("2.5e-10", u.F), "g": ("1e-06", u.S)}
        for var_name, (expected_value, expected_unit) in expected.items():
            assert self.print_rhs_of_declaration(model, var_name) == expected_value
            assert model.get_scope().resolve_to_symbol(var_name, SymbolKind.VARIABLE).get_physical_unit() == expected_unit

    def test_quantities_mixed_with_units(self, tmp_path):
        r"""Quantities are equivalent to SI units for type checking, so combining them with units of different magnitude (such as variables declared in "pA", and the predefined variable ``t`` and the function ``resolution()``, which are in "ms") is allowed."""
        fname = os.path.join(tmp_path, "model.nestml")
        with open(fname, "w") as f:
            f.write("""
model quantities_mixed_with_units_neuron:
    state:
        V_m "electrical potential" = -70 mV
        refr_t time = 0 ms

    equations:
        kernel K = (e / tau_syn) * t * exp(-t / tau_syn)
        inline I_syn "electrical current" = unit_psc * convolve(K, spikes)
        V_m' = -(V_m - E_L) / tau_m + (I_syn + I_e + I_stim) / C_m
        refr_t' = -1

    parameters:
        C_m "electrical capacitance" = 250 pF
        tau_m time = 10 ms
        tau_syn time = 2 ms
        E_L "electrical potential" = -70 mV
        I_e pA = 0 pA
        refr_T time = 2 ms

    internals:
        unit_psc "electrical current" = 1 pA
        refr_steps integer = steps(refr_T)

    input:
        spikes <- spike
        I_stim pA <- continuous

    update:
        if refr_t > resolution():
            integrate_odes(refr_t)
        else:
            integrate_odes(V_m)

        refr_t = max(refr_t - resolution(), 0 ms)
""")

        generate_target(input_path=fname, target_platform="NONE", logging_level="ERROR")
        assert len(Logger.get_messages("quantities_mixed_with_units_neuron", LoggingLevel.ERROR)) == 0

        # comparing quantities with units that differ only in magnitude (``refr_t > resolution()``) should not cause a warning
        assert not any("comparison" in msg[-1] for msg in Logger.get_messages("quantities_mixed_with_units_neuron", LoggingLevel.WARNING))

    @pytest.mark.parametrize("literal", ["-70 \"electrical potential\"", "10 time", "1 / time"])
    def test_quantity_in_numeric_literal(self, tmp_path, literal):
        r"""Quantities can only be used as data types, not as the unit of a numeric literal."""
        fname = os.path.join(tmp_path, "model.nestml")
        with open(fname, "w") as f:
            f.write("""
model quantity_in_literal_neuron:
    state:
        x real = 0

    update:
        x = """ + literal + """
""")
        generate_target(input_path=fname, target_platform="NONE", logging_level="ERROR")
        assert len(Logger.get_messages("quantity_in_literal_neuron", LoggingLevel.ERROR)) > 0

    def test_variable_with_quantity_name_in_numeric_literal(self, tmp_path):
        r"""A variable named after a quantity can still be used after a numeric literal."""
        model = self.parse_string(tmp_path, """
model variable_named_time_neuron:
    state:
        x "electrical potential" = 0 mV

    parameters:
        time real = 2

    update:
        x = 2 mV * time
""")
        model, _ = self.transform(model, NEST_UNIT_SYSTEM)
        assert self.print_rhs_of_first_assignment_in_update_block(model) == "2.0 * time"

    def test_unknown_quantity(self, tmp_path):
        fname = os.path.join(tmp_path, "model.nestml")
        with open(fname, "w") as f:
            f.write("""
model unknown_quantity_neuron:
    state:
        V_m "electrical potentiality" = -70 mV
""")
        model = ModelParser.parse_file(fname)
        assert len(Logger.get_messages(model.get_model_list()[0], LoggingLevel.ERROR)) > 0

    #
    #   time and derivatives
    #

    @pytest.mark.parametrize("unit_system, expected_V_m, expected_rate", [(NEST_UNIT_SYSTEM, "-(V_m - E_L) / tau_m + I_e / C_m", "0.001 * (-rate_inst * rate)"),
                                                                          (SI_UNIT_SYSTEM, "-(V_m - E_L) / tau_m + I_e / C_m", "-rate_inst * rate")])
    def test_derivatives(self, tmp_path, unit_system, expected_V_m, expected_rate):
        r"""The derivative of a variable is expressed in the unit of the variable divided by the unit of time of the unit system."""
        model = self.parse_string(tmp_path, """
model derivative_test_neuron:
    state:
        V_m "electrical potential" = -70 mV
        rate_inst real = 1.

    equations:
        V_m' = -(V_m - E_L) / tau_m + I_e / C_m
        rate_inst' = -rate_inst * rate

    parameters:
        C_m "electrical capacitance" = 250 pF
        tau_m time = 10 ms
        E_L "electrical potential" = -70 mV
        I_e "electrical current" = 100 pA
        rate frequency = 10 Hz
""")
        model, _ = self.transform(model, unit_system)
        assert self.print_rhs_of_ode(model, "V_m'") == expected_V_m
        assert self.print_rhs_of_ode(model, "rate_inst'") == expected_rate

    @pytest.mark.parametrize("unit_system", [NEST_UNIT_SYSTEM, SI_UNIT_SYSTEM])
    def test_implicit_cast_in_kernel_ode(self, tmp_path, unit_system):
        r"""``g$`` is declared as real, but implicitly holds a value in units of 1/[time] (the units of the unit system). No conversion factors should be inserted in the kernel ODEs."""
        model = self.parse_string(tmp_path, """
model kernel_ode_test_neuron:
    state:
        g real = 0
        g$ real = g_const * (1 / tau_rise - 1 / tau_decay)

    equations:
        kernel g' = g$ - g / tau_rise,
               g$' = -g$ / tau_decay
        inline I_syn "electrical current" = unit_psc * convolve(g, spikes)

    parameters:
        tau_rise time = 0.2 ms
        tau_decay time = 2 ms
        g_const real = 1.

    internals:
        unit_psc "electrical current" = 1 pA

    input:
        spikes <- spike
""")
        model, _ = self.transform(model, unit_system)
        kernel = model.get_equations_blocks()[0].get_kernels()[0]
        assert [self.printer.print(expr) for expr in kernel.get_expressions()] == ["g$ - g / tau_rise", "-g$ / tau_decay"]
        assert self.print_rhs_of_declaration(model, "g$") == "g_const * (1 / tau_rise - 1 / tau_decay)"

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "1000.0 * (1.0 / rate)"), (SI_UNIT_SYSTEM, "1.0 / rate")])
    def test_time_conversion_in_function_argument(self, tmp_path, unit_system, expected):
        model = self.parse_string(tmp_path, """
model steps_test_neuron:
    parameters:
        rate frequency = 10 Hz

    internals:
        n_steps integer = steps(1. / rate)
""")
        model, _ = self.transform(model, unit_system)
        function_call = model.get_internals_blocks()[0].get_declarations()[0].get_expression().get_function_call()
        assert self.printer.print(function_call.get_args()[0]) == expected

    #
    #   spike input
    #

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "unit_psc * spikes * 1.0"), (SI_UNIT_SYSTEM, "unit_psc * spikes * 1.0")])
    def test_spike_input_in_event_handler(self, tmp_path, unit_system, expected):
        r"""In an event handler, the value of a spike input port is the spike weight divided by 1 s."""
        model = self.parse_string(tmp_path, """
model spike_test_neuron:
    state:
        I_syn "electrical current" = 0 pA

    internals:
        unit_psc "electrical current" = 1 pA

    input:
        spikes <- spike

    onReceive(spikes):
        I_syn += unit_psc * spikes * 1 s
""")
        model, _ = self.transform(model, unit_system)
        assignment = model.get_on_receive_blocks()[0].get_stmts_body().get_stmts()[0].small_stmt.get_assignment()
        assert self.printer.print(assignment.get_expression()) == expected

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "unit_psp * convolve(K_delta, spikes) * (1.0)"), (SI_UNIT_SYSTEM, "unit_psp * convolve(K_delta, spikes) * (1.0)")])
    def test_spike_input_delta_kernel(self, tmp_path, unit_system, expected):
        r"""The convolution with a delta kernel is equal to the spike train multiplied by 1 ms (as ``delta()`` takes an argument in ms), so that the jump in V_m upon a spike is equal to ``unit_psp`` times the spike weight, independent of the unit system."""
        model = self.parse_string(tmp_path, """
model delta_test_neuron:
    state:
        V_m "electrical potential" = 0 mV

    equations:
        kernel K_delta = delta(t)
        V_m' = unit_psp * convolve(K_delta, spikes) * (1 ms**-1)

    internals:
        unit_psp "electrical potential" = 1 mV

    input:
        spikes <- spike
""")
        model, _ = self.transform(model, unit_system)
        assert self.print_rhs_of_ode(model, "V_m'") == expected

    @pytest.mark.parametrize("unit_system, expected", [(NEST_UNIT_SYSTEM, "unit_psc * spikes"), (SI_UNIT_SYSTEM, "unit_psc * spikes")])
    def test_spike_input_in_ode(self, tmp_path, unit_system, expected):
        r"""In an ODE, the spike input port represents a train of delta pulses, with units of 1/s."""
        model = self.parse_string(tmp_path, """
model spike_ode_test_neuron:
    state:
        I_syn "electrical current" = 0 pA

    equations:
        I_syn' = -I_syn / tau_syn + unit_psc * spikes

    parameters:
        tau_syn time = 2 ms

    internals:
        unit_psc "electrical current" = 1 pA

    input:
        spikes <- spike
""")
        model, _ = self.transform(model, unit_system)
        assert self.print_rhs_of_ode(model, "I_syn'") == "-I_syn / tau_syn + " + expected

    @pytest.mark.parametrize("unit_system", [NEST_UNIT_SYSTEM, SI_UNIT_SYSTEM])
    def test_spike_input_added_in_ode(self, tmp_path, unit_system):
        r"""A dimensionless variable that is incremented by the spike weight upon each spike."""
        model = self.parse_string(tmp_path, """
model spike_ode_test_neuron:
    state:
        x real = 0

    equations:
        x' = -x / tau + spikes

    parameters:
        tau time = 2 ms

    input:
        spikes[2] <- spike
""")
        model, _ = self.transform(model, unit_system)
        assert self.print_rhs_of_ode(model, "x'") == "-x / tau + spikes"

    #
    #   unit systems
    #

    def test_unit_system_target_units(self):
        nest_unit_system = UnitSystem.from_json_file(NEST_UNIT_SYSTEM)
        si_unit_system = UnitSystem.from_json_file(SI_UNIT_SYSTEM)

        assert nest_unit_system.get_time_unit() == u.ms
        assert si_unit_system.get_time_unit() == u.s

        for unit, expected_nest, expected_si in [(u.V, u.mV, u.V),
                                                 (u.uS, u.nS, u.S),
                                                 (u.mV / u.ms, u.mV / u.ms, u.V / u.s),
                                                 (u.nS * u.mV, u.pA, u.A),
                                                 (u.pF / u.mV / u.ms, u.pF / u.mV / u.ms, u.F / u.V / u.s),
                                                 (u.ms / u.s, u.dimensionless_unscaled, u.dimensionless_unscaled)]:
            assert nest_unit_system.get_target_unit(unit) == expected_nest
            assert si_unit_system.get_target_unit(unit) == expected_si

    def test_unit_system_from_dict(self):
        unit_system = UnitSystem.from_option({"electrical potential": "uV", "time": "s"})
        assert unit_system.get_target_unit(u.mV) == u.uV
        assert unit_system.get_target_unit(u.mV / u.ms) == u.uV / u.s

    def test_unit_system_codegen_option(self, tmp_path):
        r"""The unit system can be selected by means of the ``unit_system`` code generator option."""
        fname = os.path.join(tmp_path, "model.nestml")
        with open(fname, "w") as f:
            f.write("""
model unit_system_option_test_neuron:
    state:
        V_m "electrical potential" = -70 mV
""")

        generate_target(input_path=fname, target_platform="NONE", logging_level="ERROR", codegen_opts={"unit_system": {"electrical potential": "uV", "time": "s"}})
        assert len(Logger.get_messages("unit_system_option_test_neuron", LoggingLevel.ERROR)) == 0

    def test_invalid_unit_system(self):
        with pytest.raises(UnitSystemException):
            UnitSystem({"electrical potential": "ms"})

        with pytest.raises(UnitSystemException):
            UnitSystem({"electrical potentiality": "mV"})
