# -*- coding: utf-8 -*-
#
# non_dimensionalisation_transformer.py
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

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

try:
    # Available in the standard library starting with Python 3.12
    from typing import override
except ImportError:
    # Fallback for Python 3.8 - 3.11
    from typing_extensions import override

import os

from astropy import units as u

from pynestml.meta_model.ast_arithmetic_operator import ASTArithmeticOperator
from pynestml.meta_model.ast_assignment import ASTAssignment
from pynestml.meta_model.ast_bit_operator import ASTBitOperator
from pynestml.meta_model.ast_block_with_variables import ASTBlockWithVariables
from pynestml.meta_model.ast_comparison_operator import ASTComparisonOperator
from pynestml.meta_model.ast_data_type import ASTDataType
from pynestml.meta_model.ast_declaration import ASTDeclaration
from pynestml.meta_model.ast_equations_block import ASTEquationsBlock
from pynestml.meta_model.ast_expression import ASTExpression
from pynestml.meta_model.ast_function import ASTFunction
from pynestml.meta_model.ast_function_call import ASTFunctionCall
from pynestml.meta_model.ast_inline_expression import ASTInlineExpression
from pynestml.meta_model.ast_input_block import ASTInputBlock
from pynestml.meta_model.ast_kernel import ASTKernel
from pynestml.meta_model.ast_logical_operator import ASTLogicalOperator
from pynestml.meta_model.ast_model import ASTModel
from pynestml.meta_model.ast_node_factory import ASTNodeFactory
from pynestml.meta_model.ast_ode_equation import ASTOdeEquation
from pynestml.meta_model.ast_on_condition_block import ASTOnConditionBlock
from pynestml.meta_model.ast_on_receive_block import ASTOnReceiveBlock
from pynestml.meta_model.ast_simple_expression import ASTSimpleExpression
from pynestml.meta_model.ast_stmts_body import ASTStmtsBody
from pynestml.meta_model.ast_update_block import ASTUpdateBlock
from pynestml.meta_model.ast_variable import ASTVariable
from pynestml.symbols.integer_type_symbol import IntegerTypeSymbol
from pynestml.symbols.predefined_functions import PredefinedFunctions
from pynestml.symbols.predefined_types import PredefinedTypes
from pynestml.symbols.real_type_symbol import RealTypeSymbol
from pynestml.symbols.symbol import SymbolKind
from pynestml.symbols.template_type_symbol import TemplateTypeSymbol
from pynestml.symbols.type_symbol import TypeSymbol
from pynestml.symbols.unit_type_symbol import UnitTypeSymbol
from pynestml.symbols.variadic_type_symbol import VariadicTypeSymbol
from pynestml.transformers.transformer import Transformer
from pynestml.utils.ast_utils import ASTUtils
from pynestml.utils.logger import Logger, LoggingLevel
from pynestml.utils.unit_system import UnitSystem
from pynestml.visitors.ast_data_type_visitor import ASTDataTypeVisitor
from pynestml.visitors.ast_parent_visitor import ASTParentVisitor
from pynestml.visitors.ast_symbol_table_visitor import ASTSymbolTableVisitor


ExpressionNode = Union[ASTExpression, ASTSimpleExpression]


class NonDimensionalisationTransformer(Transformer):
    r"""Convert a model in which quantities are expressed in terms of physical units into a model that contains only real-typed (dimensionless) quantities.

    After this transformation, every numerical value in the model is expressed in the unit that the selected *unit system* specifies for its quantity (for instance, millivolts for voltages and milliseconds for time in NEST Simulator, or volts and seconds when using SI units). All conversion factors that are needed to combine values with different units are inserted explicitly into the expressions of the model, as multiplications by real-valued constants. All data types that refer to physical units or quantities are replaced by ``real``. As a result, printers and code generators do not have to deal with physical units or conversion factors at all.

    For example, with a unit system that specifies ``"time": "ms"``, ``"electrical potential": "mV"`` and ``"electrical current": "pA"``,

    .. code-block:: nestml

       parameters:
           tau time = 0.01 s
           V_th "electrical potential" = -55 mV
           I_e "electrical current" = 1 nA

    is transformed into

    .. code-block:: nestml

       parameters:
           tau real = 10.0
           V_th real = -55.0
           I_e real = 1000.0

    The unit in which each variable is expressed after the transformation is retained in the ``physical_unit`` attribute of the corresponding ``ASTDataType`` (and of the variable symbol, see :py:meth:`VariableSymbol.get_physical_unit`), so that it can, for instance, be printed in code comments. Additionally, a copy of the model as it was before the transformation is stored in the metadata of the model, under the key ``"original_model"``.

    Time is handled like any other quantity. The unit system specifies the unit of time, which must be the unit used by the simulation platform; for example, the values of ``t``, ``resolution()`` and ``timestep()`` are expressed in this unit. The derivative of a variable ``x`` is expressed in the unit of ``x`` divided by the unit of time.

    Spiking input ports are a special case. Inside ``onReceive`` blocks and other statements, the value of a spike input port is (by convention) taken to be the weight of the incoming spike(s) divided by 1 s. Inside the equations block, a spike input port represents a train of Dirac delta pulses, the area of each of which is equal to its weight; the code generator processes these as instantaneous jumps (see ``ConvolutionsToBuffersTransformer``). The convolution of a delta-function kernel ``delta(t)`` with a spike input port is dimensionless; as the parameter of ``delta()`` is in ms, this means that the delta pulse is scaled by 1 ms.

    Options
    -------

    - **unit_system**: The unit system to use. Either the path to a JSON file, or a dictionary, that maps the names of quantities to the names of units, for example: ``{"time": "ms", "electrical potential": "mV"}``. The names of quantities are those defined by astropy (see https://docs.astropy.org/en/stable/units/physical_types.html). Quantities that are not specified are expressed by combining the units of their constituent quantities (for example, "electrical potential / time" would become mV/ms), or ultimately in SI units. Each target platform comes with its own default unit system (see ``unit_system.json`` in the target's resources directory).
    """

    _default_options = {
        "unit_system": None
    }

    def __init__(self, options: Optional[Mapping[str, Any]] = None):
        super(Transformer, self).__init__(options)

    def get_unit_system(self) -> UnitSystem:
        unit_system = self.get_option("unit_system")
        if unit_system is None:
            # default to SI units
            return UnitSystem({}, name="SI")

        return UnitSystem.from_option(unit_system)

    @override
    def transform(self,
                  models: Iterable[ASTModel],
                  metadata: Dict[str, Dict[str, Any]]) -> Iterable[ASTModel]:
        unit_system = self.get_unit_system()
        Logger.log_message(None, None, "Non-dimensionalising models using unit system: " + str(unit_system), None, LoggingLevel.INFO)

        for model in models:
            if not model.name in metadata.keys():
                metadata[model.name] = {}

            original_model = model.clone()
            original_model.parent_ = None
            original_model.accept(ASTParentVisitor())
            metadata[model.name]["original_model"] = original_model
            metadata[model.name]["unit_system"] = unit_system

            ModelNonDimensionaliser(model, unit_system).process()

            model.accept(ASTParentVisitor())
            model.accept(ASTSymbolTableVisitor())

        return models


class ModelNonDimensionaliser:
    r"""
    Helper class that performs the non-dimensionalisation of a single model (see :py:class:`NonDimensionalisationTransformer`).

    The central concept is the *carried unit* of an expression: the unit in which the numerical value of the (rewritten) expression is expressed. For example, after rewriting, a reference to a variable ``V_m`` (declared as an electrical potential) has carried unit "mV" in the NEST unit system, and the expression ``V_m / tau_m`` has carried unit "mV/ms". Wherever the value of an expression is "consumed", for instance on the right-hand side of an assignment, its carried unit is converted to the unit of the receiving variable by multiplying by an explicit conversion factor (if this factor is not equal to 1).
    """

    def __init__(self, model: ASTModel, unit_system: UnitSystem):
        self._model = model
        self._unit_system = unit_system
        self._time_unit = unit_system.get_time_unit()
        self._in_equations_block = False
        self._function_return_unit_stack: List[Optional[u.UnitBase]] = []

    def process(self) -> None:
        r"""Non-dimensionalise the model in-place."""
        model = self._model

        # make sure that all symbols and types are up to date
        model.accept(ASTParentVisitor())
        model.accept(ASTSymbolTableVisitor())

        for block in model.get_body().get_body_elements():
            if isinstance(block, ASTBlockWithVariables):
                for decl in block.get_declarations():
                    self._process_declaration(decl)

            elif isinstance(block, ASTEquationsBlock):
                self._in_equations_block = True
                for decl in block.get_declarations():
                    if isinstance(decl, ASTInlineExpression):
                        self._process_inline_expression(decl)
                    elif isinstance(decl, ASTOdeEquation):
                        self._process_ode_equation(decl)
                    elif isinstance(decl, ASTKernel):
                        self._process_kernel(decl)
                self._in_equations_block = False

            elif isinstance(block, ASTInputBlock):
                for port in block.get_input_ports():
                    if port.has_size_parameter():
                        port.size_parameter = self._process_and_convert(port.get_size_parameter(), u.dimensionless_unscaled)

                    if port.has_datatype():
                        self._non_dimensionalise_data_type(port.get_datatype())

            elif isinstance(block, ASTFunction):
                self._process_function(block)

            elif isinstance(block, ASTUpdateBlock):
                self._process_stmts_body(block.get_stmts_body())

            elif isinstance(block, ASTOnReceiveBlock):
                self._process_stmts_body(block.get_stmts_body())

            elif isinstance(block, ASTOnConditionBlock):
                block.cond_expr, _ = self._process_expression(block.get_cond_expr())
                self._process_stmts_body(block.get_stmts_body())

    #
    #   units
    #

    def _get_unit_of_type_symbol(self, type_symbol: Optional[TypeSymbol]) -> Optional[u.UnitBase]:
        r"""Returns the physical unit corresponding to the type symbol; "dimensionless" for real and integer types, and None for non-numeric types."""
        if isinstance(type_symbol, UnitTypeSymbol):
            return u.Unit(type_symbol.astropy_unit)

        if isinstance(type_symbol, (RealTypeSymbol, IntegerTypeSymbol)):
            return u.dimensionless_unscaled

        return None

    def _get_unit_of_data_type(self, data_type: Optional[ASTDataType]) -> Optional[u.UnitBase]:
        r"""Returns the physical unit corresponding to the data type; "dimensionless" for real and integer types, and None for non-numeric types."""
        if data_type is None:
            return None

        data_type.accept(ASTDataTypeVisitor())    # make sure the type symbol is up to date
        return self._get_unit_of_type_symbol(data_type.get_type_symbol())

    def _get_target_unit(self, unit: Optional[u.UnitBase], differential_order: int = 0) -> Optional[u.UnitBase]:
        r"""
        Returns the unit in which the numerical value of a variable is expressed after the transformation.

        For a variable ``x`` with unit ``U``, this is the unit specified by the unit system for ``U``, for instance, "mV" for "V". For the derivative ``x'`` (``differential_order = 1``), it is the unit of ``x`` divided by the unit system's time unit, and so on for higher orders.

        :param unit: the unit as declared (for a derivative, the unit of the derivative, e.g. "V/s" for ``V_m'``)
        :param differential_order: the differential order of the variable
        """
        if unit is None:
            return None

        if differential_order == 0:
            return self._unit_system.get_target_unit(unit)

        base_unit = u.Unit(unit * u.s**differential_order)    # unit of the variable of differential order 0
        return self._unit_system.get_target_unit(base_unit) / self._time_unit**differential_order

    def _resolve_variable(self, variable: ASTVariable) -> Tuple[Optional[Any], int]:
        r"""
        Resolve the symbol for the given variable. Returns a tuple of the symbol and the differential order of the variable relative to the symbol. For instance, for the variable ``V_m'``, if there is a symbol named ``V_m'``, return that symbol with order 0. If there is only a symbol named ``V_m``, return that symbol and order 1.
        """
        scope = variable.get_scope()
        if scope is None:
            return None, 0

        symbol = scope.resolve_to_symbol(variable.get_complete_name(), SymbolKind.VARIABLE)
        if symbol is not None:
            return symbol, 0

        symbol = scope.resolve_to_symbol(variable.get_name(), SymbolKind.VARIABLE)
        if symbol is not None:
            return symbol, variable.get_differential_order()

        return None, 0

    def _get_target_unit_of_variable(self, variable: ASTVariable) -> Optional[u.UnitBase]:
        r"""Returns the unit in which the numerical value of the given variable is expressed after the transformation."""
        symbol, relative_order = self._resolve_variable(variable)
        if symbol is None:
            return None

        unit = self._get_unit_of_type_symbol(symbol.get_type_symbol())
        if unit is None:
            return None

        symbol_order = symbol.get_symbol_name().count("'")
        if relative_order > 0:
            unit = unit / u.s**relative_order

        return self._get_target_unit(unit, symbol_order + relative_order)

    def _get_carried_unit_of_variable_reference(self, variable: ASTVariable) -> Optional[u.UnitBase]:
        r"""Returns the carried unit of a reference to the given variable in an expression."""
        symbol, _ = self._resolve_variable(variable)

        if symbol is not None and symbol.is_spike_input_port():
            if self._in_equations_block:
                # spike train (sum of Dirac delta pulses): the numerical value is in units of 1 / [unit of time]
                return self._time_unit**-1

            # in event handlers: the value of the port is the spike weight divided by 1 s
            return u.s**-1

        return self._get_target_unit_of_variable(variable)

    #
    #   expression rewriting
    #

    def _make_numeric_literal(self, value: float, source_node) -> ASTSimpleExpression:
        node = ASTNodeFactory.create_ast_simple_expression(numeric_literal=value, source_position=source_node.get_source_position())
        node.update_scope(source_node.get_scope())
        return node

    def _apply_factor(self, node: ExpressionNode, factor: float) -> ExpressionNode:
        r"""Return an expression that is equal to ``factor * node``."""
        if UnitSystem.is_unity(factor):
            return node

        if isinstance(node, ASTSimpleExpression) and node.is_numeric_literal():
            # fold the factor into the literal
            node.set_numeric_literal(UnitSystem.round_factor(float(node.get_numeric_literal()) * factor))
            return node

        if isinstance(node, ASTSimpleExpression) and node.is_inf_literal:
            return node

        if isinstance(node, ASTExpression) and node.is_unary_operator() and node.get_unary_operator().is_unary_minus \
           and isinstance(node.get_expression(), ASTSimpleExpression) and node.get_expression().is_numeric_literal():
            # fold the factor into the literal, e.g. -70 mV
            node.expression = self._apply_factor(node.get_expression(), factor)
            return node

        if isinstance(node, ASTSimpleExpression) or (isinstance(node, ASTExpression) and node.is_encapsulated):
            operand = node
        else:
            operand = ASTNodeFactory.create_ast_expression(is_encapsulated=True, expression=node, source_position=node.get_source_position())
            operand.update_scope(node.get_scope())

        times_op = ASTNodeFactory.create_ast_arithmetic_operator(is_times_op=True, source_position=node.get_source_position())
        new_node = ASTNodeFactory.create_ast_compound_expression(lhs=self._make_numeric_literal(factor, node),
                                                                 binary_operator=times_op,
                                                                 rhs=operand,
                                                                 source_position=node.get_source_position())
        new_node.update_scope(node.get_scope())
        return new_node

    def _convert(self, node: ExpressionNode, from_unit: Optional[u.UnitBase], to_unit: Optional[u.UnitBase]) -> ExpressionNode:
        r"""
        Convert the expression from ``from_unit`` to ``to_unit`` by multiplying by a constant factor.

        If the units have different physical dimensions, no conversion is performed. This can only be the case if the type checker has already allowed this as an implicit cast, for instance, when assigning a quantity with a physical unit to a variable of type ``real``; in this case, the numerical value is interpreted as being expressed in the units of the unit system.
        """
        if from_unit is None or to_unit is None:
            return node

        factor = UnitSystem.get_conversion_factor(from_unit, to_unit)
        if factor is None:
            # different dimensions: implicit cast
            return node

        return self._apply_factor(node, factor)

    def _normalise(self, node: ExpressionNode, unit: Optional[u.UnitBase]) -> Tuple[ExpressionNode, Optional[u.UnitBase]]:
        r"""Convert the expression into the unit system's unit for its quantity, e.g. from "V" into "mV"."""
        if unit is None:
            return node, None

        target_unit = self._unit_system.get_target_unit(unit)
        return self._convert(node, unit, target_unit), target_unit

    def _process_and_convert(self, node: Optional[ExpressionNode], target_unit: Optional[u.UnitBase]) -> Optional[ExpressionNode]:
        r"""Rewrite an expression, and convert its value into ``target_unit``."""
        if node is None:
            return None

        node, carried_unit = self._process_expression(node)
        return self._convert(node, carried_unit, target_unit)

    def _process_expression(self, node: ExpressionNode) -> Tuple[ExpressionNode, Optional[u.UnitBase]]:
        r"""
        Rewrite an expression in-place, removing all physical units, and inserting conversion factors where necessary. Returns the rewritten expression (which may be a different node than the one passed in) and its carried unit (None for non-numeric expressions like booleans and strings, or for values that are scale-free, like ``inf``).
        """
        if isinstance(node, ASTSimpleExpression):
            return self._process_simple_expression(node)

        assert isinstance(node, ASTExpression)

        if node.is_encapsulated:
            node.expression, unit = self._process_expression(node.get_expression())
            return node, unit

        if node.is_logical_not:
            node.expression, _ = self._process_expression(node.get_expression())
            return node, None

        if node.is_unary_operator():
            node.expression, unit = self._process_expression(node.get_expression())
            if node.get_unary_operator().is_unary_tilde:
                return node, u.dimensionless_unscaled

            return node, unit

        if node.is_ternary_operator():
            node.condition, _ = self._process_expression(node.get_condition())
            node.if_true, unit_true = self._process_expression(node.get_if_true())
            node.if_not, unit_not = self._process_expression(node.get_if_not())
            return self._normalise_pair(node, "if_true", unit_true, "if_not", unit_not)

        if node.is_compound_expression():
            op = node.get_binary_operator()

            if isinstance(op, ASTArithmeticOperator) and op.is_pow_op:
                return self._process_power(node)

            node.lhs, lhs_unit = self._process_expression(node.get_lhs())
            node.rhs, rhs_unit = self._process_expression(node.get_rhs())

            if isinstance(op, ASTArithmeticOperator):
                if op.is_times_op:
                    return node, self._multiply_units(lhs_unit, rhs_unit)

                if op.is_div_op:
                    return node, self._divide_units(lhs_unit, rhs_unit)

                if op.is_integer_div_op:
                    if lhs_unit is not None and rhs_unit is not None and UnitSystem.get_conversion_factor(lhs_unit, rhs_unit) is not None:
                        # make sure the integer division is carried out on values that have the same scale
                        node, _ = self._normalise_pair(node, "lhs", lhs_unit, "rhs", rhs_unit)
                        return node, u.dimensionless_unscaled

                    return node, self._divide_units(lhs_unit, rhs_unit)

                # plus, minus, modulo
                return self._normalise_pair(node, "lhs", lhs_unit, "rhs", rhs_unit)

            if isinstance(op, ASTComparisonOperator):
                node, _ = self._normalise_pair(node, "lhs", lhs_unit, "rhs", rhs_unit)
                return node, None

            if isinstance(op, ASTLogicalOperator):
                return node, None

            # bit operator
            assert isinstance(op, ASTBitOperator)
            return node, u.dimensionless_unscaled

        if node.get_expression() is not None:
            node.expression, unit = self._process_expression(node.get_expression())
            return node, unit

        raise Exception("Unknown expression type: " + str(node))

    def _multiply_units(self, lhs_unit, rhs_unit):
        if lhs_unit is None:
            return rhs_unit

        if rhs_unit is None:
            return lhs_unit

        return u.Unit(lhs_unit * rhs_unit)

    def _divide_units(self, lhs_unit, rhs_unit):
        if rhs_unit is None:
            return lhs_unit

        if lhs_unit is None:
            return u.Unit(rhs_unit**-1)

        return u.Unit(lhs_unit / rhs_unit)

    def _normalise_pair(self, node: ASTExpression, lhs_attr: str, lhs_unit, rhs_attr: str, rhs_unit) -> Tuple[ASTExpression, Optional[u.UnitBase]]:
        r"""
        For operators that require both operands to have the same unit (addition, subtraction, comparison, etc.): if the operands are expressed in different units, convert both into the unit system's unit for their quantity.

        If the operands have different physical dimensions (which can only occur in the case of implicit casts, for instance, when adding a ``real`` to a variable with units, as in ``V_m + 1``), no conversion takes place: the dimensionless operand is interpreted as being expressed in the unit of the other operand.
        """
        if lhs_unit is None and rhs_unit is None:
            return node, None

        if lhs_unit is None:
            return node, rhs_unit

        if rhs_unit is None:
            return node, lhs_unit

        factor = UnitSystem.get_conversion_factor(rhs_unit, lhs_unit)

        if factor is None:
            # different physical dimensions (implicit cast): no conversion
            if lhs_unit.physical_type == "dimensionless":
                return node, rhs_unit

            return node, lhs_unit

        if UnitSystem.is_unity(factor):
            # both operands are already expressed in the same unit; no conversion necessary
            return node, lhs_unit

        target_unit = self._unit_system.get_target_unit(lhs_unit)
        setattr(node, lhs_attr, self._convert(getattr(node, lhs_attr), lhs_unit, target_unit))
        setattr(node, rhs_attr, self._convert(getattr(node, rhs_attr), rhs_unit, target_unit))

        return node, target_unit

    def _get_constant_value(self, node: ExpressionNode) -> Optional[float]:
        r"""If the expression is a numerical constant (for instance, ``2`` or ``-(1/2)``), return its value; otherwise, return None."""
        if isinstance(node, ASTSimpleExpression):
            if node.is_numeric_literal() and not node.is_numeric_literal_with_unit():
                return float(node.get_numeric_literal())

            return None

        if node.is_encapsulated:
            return self._get_constant_value(node.get_expression())

        if node.is_unary_operator():
            value = self._get_constant_value(node.get_expression())
            if value is None:
                return None

            if node.get_unary_operator().is_unary_minus:
                return -value

            if node.get_unary_operator().is_unary_plus:
                return value

            return None

        if node.is_compound_expression() and isinstance(node.get_binary_operator(), ASTArithmeticOperator):
            lhs = self._get_constant_value(node.get_lhs())
            rhs = self._get_constant_value(node.get_rhs())
            if lhs is None or rhs is None:
                return None

            op = node.get_binary_operator()
            if op.is_times_op:
                return lhs * rhs

            if op.is_div_op and not rhs == 0:
                return lhs / rhs

            if op.is_plus_op:
                return lhs + rhs

            if op.is_minus_op:
                return lhs - rhs

            if op.is_pow_op:
                return lhs ** rhs

        return None

    def _process_power(self, node: ASTExpression) -> Tuple[ASTExpression, Optional[u.UnitBase]]:
        r"""Rewrite ``base ** exponent``."""
        node.lhs, base_unit = self._process_expression(node.get_lhs())
        node.rhs, exponent_unit = self._process_expression(node.get_rhs())
        node.lhs, node.rhs, unit = self._convert_power_operands(node.lhs, base_unit, node.rhs, exponent_unit)

        return node, unit

    def _convert_power_operands(self, base: ExpressionNode, base_unit: Optional[u.UnitBase], exponent: ExpressionNode, exponent_unit: Optional[u.UnitBase]) -> Tuple[ExpressionNode, ExpressionNode, Optional[u.UnitBase]]:
        r"""For a power ``base ** exponent`` (or ``pow(base, exponent)``), convert the operands where necessary, and return them, together with the carried unit of the result."""
        # the exponent must be dimensionless
        exponent = self._convert(exponent, exponent_unit, u.dimensionless_unscaled)

        if base_unit is None:
            return base, exponent, None

        if base_unit.physical_type == "dimensionless":
            # make sure the base is unscaled (e.g. not in ms/s)
            return self._convert(base, base_unit, u.dimensionless_unscaled), exponent, u.dimensionless_unscaled

        exponent_value = self._get_constant_value(exponent)
        if exponent_value is None:
            # the exponent cannot be determined at code generation time; express the base in the unit system's unit for its quantity
            base, base_unit = self._normalise(base, base_unit)
            Logger.log_message(None, None, "The exponent of a power with a dimensional base could not be determined at code generation time; the result is assumed to have units of " + str(base_unit), base.get_source_position(), LoggingLevel.WARNING)
            return base, exponent, base_unit

        if float(exponent_value).is_integer():
            exponent_value = int(exponent_value)

        return base, exponent, u.Unit(base_unit ** exponent_value)

    def _process_simple_expression(self, node: ASTSimpleExpression) -> Tuple[ExpressionNode, Optional[u.UnitBase]]:
        if node.is_function_call():
            return self._process_function_call(node)

        if node.is_inf_literal:
            return node, None

        if node.is_string() or node.is_boolean_true or node.is_boolean_false:
            return node, None

        if node.is_numeric_literal():
            if node.is_numeric_literal_with_unit():
                # numeric literal with a unit, e.g. "10 mV": remove the unit; the carried unit is the unit of the literal
                node.unitType.accept(ASTDataTypeVisitor())    # make sure the type symbol is up to date
                unit = self._get_unit_of_type_symbol(node.unitType.get_type_symbol())
                node.unitType = None
                node.set_variable(None)
                node.set_numeric_literal(float(node.get_numeric_literal()))    # make sure the literal is a float, so that integer division does not occur in the generated code (e.g. for "1 / 2 ms")
                return node, unit

            return node, u.dimensionless_unscaled

        if node.is_variable() or node.is_delay_variable():
            variable = node.get_variable()
            if variable.get_scope() is None:
                if node.get_scope() is None:
                    return node, None

                variable.update_scope(node.get_scope())

            if variable.has_vector_parameter():
                variable.vector_parameter = self._process_and_convert(variable.get_vector_parameter(), u.dimensionless_unscaled)

            symbol, _ = self._resolve_variable(variable)
            if symbol is None:
                type_symbol = variable.get_scope().resolve_to_symbol(variable.get_complete_name(), SymbolKind.TYPE)
                if type_symbol is not None:
                    # a physical unit used as a variable, e.g. "mV" in "V_m / mV": replace it with the numeric literal "1"
                    unit = self._get_unit_of_type_symbol(type_symbol)
                    node.set_variable(None)
                    node.set_numeric_literal(1.)
                    return node, unit

                return node, None

            return node, self._get_carried_unit_of_variable_reference(variable)

        return node, None

    def _process_function_call(self, node: ASTSimpleExpression) -> Tuple[ExpressionNode, Optional[u.UnitBase]]:
        function_call: ASTFunctionCall = node.get_function_call()
        function_name = function_call.get_name()

        if function_name == PredefinedFunctions.INTEGRATE_ODES:
            # the arguments are names of state variables
            return node, None

        if function_name == PredefinedFunctions.CONVOLVE:
            # the arguments are a kernel and a spike input port
            return node, self._get_carried_unit_of_convolution(function_call)

        function_symbol = function_call.get_scope().resolve_to_symbol(function_name, SymbolKind.FUNCTION)

        if function_symbol is None:
            if ASTUtils.is_function_delay_variable(function_call):
                # delay variable, e.g. ``V_m(t - delay)``; the argument is left as-is
                delay_variable_symbol = ASTUtils.get_delay_variable_symbol(function_call)
                return node, self._get_target_unit(self._get_unit_of_type_symbol(delay_variable_symbol.get_type_symbol()))

            return node, None

        param_types = function_symbol.get_parameter_types()
        args = function_call.get_args()
        is_variadic = len(param_types) == 1 and isinstance(param_types[0], VariadicTypeSymbol)

        # process the arguments
        arg_units = []
        for i, arg in enumerate(args):
            args[i], arg_unit = self._process_expression(arg)
            arg_units.append(arg_unit)

        if function_name == PredefinedFunctions.POW and len(args) == 2:
            args[0], args[1], unit = self._convert_power_operands(args[0], arg_units[0], args[1], arg_units[1])
            return node, unit

        # convert arguments into the expected units
        template_units = {}
        for i, arg in enumerate(args):
            if is_variadic:
                # unknown parameter type: express the argument in the unit system's unit for its quantity
                args[i], arg_units[i] = self._normalise(arg, arg_units[i])
                continue

            if i >= len(param_types):
                continue

            param_type = param_types[i]
            if isinstance(param_type, TemplateTypeSymbol):
                if arg_units[i] is None:
                    continue

                template_id = param_type._i
                if not template_id in template_units.keys():
                    template_units[template_id] = self._unit_system.get_target_unit(arg_units[i])

                args[i] = self._convert(arg, arg_units[i], template_units[template_id])
                continue

            param_unit = self._get_unit_of_type_symbol(param_type)
            args[i] = self._convert(arg, arg_units[i], self._get_target_unit(param_unit))

        # determine the carried unit of the return value
        return_type = function_symbol.get_return_type()

        if isinstance(return_type, TemplateTypeSymbol):
            if return_type._i in template_units.keys():
                return node, template_units[return_type._i]

            return node, None

        return node, self._get_target_unit(self._get_unit_of_type_symbol(return_type))

    def _get_carried_unit_of_convolution(self, function_call: ASTFunctionCall) -> Optional[u.UnitBase]:
        r"""
        Returns the carried unit of ``convolve(kernel, spike_input_port)``.

        For a kernel that is a function of time, the convolution with a spike train is the sum of the kernel function for each spike, weighted by the spike weight; hence, it has the same carried unit as the kernel function itself, which is dimensionless (the type of the convolution, as determined by the type checker, is always dimensionless).

        For a delta kernel ``delta(t)``, the convolution is equal to the spike train, which has the numerical value 1 / [unit of time] for every spike (see :py:meth:`_get_carried_unit_of_variable_reference`). However, because the parameter of ``delta()`` has units of ms, the convolution is dimensionless: it is equal to the spike train multiplied by 1 ms.
        """
        args = function_call.get_args()
        for arg in args:
            if not isinstance(arg, ASTSimpleExpression) or not arg.is_variable():
                continue

            kernel = self._model.get_kernel_by_name(arg.get_variable().get_name())
            if kernel is not None and ASTUtils.is_delta_kernel(kernel):
                return u.Unit(u.ms / self._time_unit)

        return u.dimensionless_unscaled

    #
    #   data types
    #

    def _non_dimensionalise_data_type(self, data_type: Optional[ASTDataType], differential_order: int = 0) -> None:
        r"""Replace a data type that refers to a physical unit or quantity by ``real``, and store the unit in which the numerical value is expressed in the ``physical_unit`` attribute."""
        if data_type is None or not data_type.is_unit_type():
            return

        unit = self._get_unit_of_data_type(data_type)
        target_unit = self._get_target_unit(unit, differential_order)

        data_type.unit_type = None
        data_type.is_real = True
        data_type.type_symbol = PredefinedTypes.get_real_type()
        if target_unit is not None and not target_unit.physical_type == "dimensionless":
            data_type.physical_unit = target_unit

    #
    #   declarations, equations, statements
    #

    def _process_declaration(self, decl: ASTDeclaration) -> None:
        variables = decl.get_variables()
        target_unit = self._get_target_unit_of_variable(variables[0]) if variables else None

        for var in variables:
            if var.has_vector_parameter():
                var.vector_parameter = self._process_and_convert(var.get_vector_parameter(), u.dimensionless_unscaled)

        if decl.size_parameter is not None:
            decl.size_parameter = self._process_and_convert(decl.size_parameter, u.dimensionless_unscaled)

        if decl.has_expression():
            decl.expression = self._process_and_convert(decl.get_expression(), target_unit)

        if decl.has_invariant():
            decl.invariant, _ = self._process_expression(decl.get_invariant())

        differential_order = variables[0].get_differential_order() if variables else 0
        self._non_dimensionalise_data_type(decl.get_data_type(), differential_order)

    def _process_inline_expression(self, inline_expr: ASTInlineExpression) -> None:
        target_unit = self._get_target_unit(self._get_unit_of_data_type(inline_expr.get_data_type()))
        inline_expr.expression = self._process_and_convert(inline_expr.get_expression(), target_unit)
        self._non_dimensionalise_data_type(inline_expr.get_data_type())

    def _process_ode_equation(self, ode_eq: ASTOdeEquation) -> None:
        target_unit = self._get_target_unit_of_variable(ode_eq.get_lhs())
        ode_eq.rhs = self._process_and_convert(ode_eq.get_rhs(), target_unit)

    def _process_kernel(self, kernel: ASTKernel) -> None:
        for i, (var, expr) in enumerate(zip(kernel.get_variables(), kernel.get_expressions())):
            if var.get_differential_order() == 0:
                # kernel as a direct function of time, e.g. ``exp(-t / tau)``: no conversion
                kernel.get_expressions()[i], _ = self._process_expression(expr)
            else:
                # kernel as ODE
                kernel.get_expressions()[i] = self._process_and_convert(expr, self._get_target_unit_of_variable(var))

    def _process_function(self, function: ASTFunction) -> None:
        return_unit = None
        if function.has_return_type():
            return_unit = self._get_target_unit(self._get_unit_of_data_type(function.get_return_type()))

        self._function_return_unit_stack.append(return_unit)
        self._process_stmts_body(function.get_stmts_body())
        self._function_return_unit_stack.pop()

        for param in function.get_parameters():
            self._non_dimensionalise_data_type(param.get_data_type())

        if function.has_return_type():
            self._non_dimensionalise_data_type(function.get_return_type())

    def _process_stmts_body(self, stmts_body: Optional[ASTStmtsBody]) -> None:
        if stmts_body is None:
            return

        for stmt in stmts_body.get_stmts():
            if stmt.is_small_stmt():
                self._process_small_stmt(stmt.small_stmt)
            else:
                self._process_compound_stmt(stmt.compound_stmt)

    def _process_small_stmt(self, small_stmt) -> None:
        if small_stmt.is_assignment():
            self._process_assignment(small_stmt.get_assignment())

        elif small_stmt.is_function_call():
            dummy = ASTNodeFactory.create_ast_simple_expression(function_call=small_stmt.get_function_call())
            self._process_function_call(dummy)

        elif small_stmt.is_declaration():
            self._process_declaration(small_stmt.get_declaration())

        elif small_stmt.is_return_stmt():
            return_stmt = small_stmt.get_return_stmt()
            if return_stmt.has_expression():
                return_unit = self._function_return_unit_stack[-1] if self._function_return_unit_stack else None
                return_stmt.expression = self._process_and_convert(return_stmt.get_expression(), return_unit)

    def _process_compound_stmt(self, compound_stmt) -> None:
        if compound_stmt.is_if_stmt():
            if_stmt = compound_stmt.get_if_stmt()
            if_clause = if_stmt.get_if_clause()
            if_clause.condition, _ = self._process_expression(if_clause.get_condition())
            self._process_stmts_body(if_clause.get_stmts_body())
            for elif_clause in if_stmt.get_elif_clauses():
                elif_clause.condition, _ = self._process_expression(elif_clause.get_condition())
                self._process_stmts_body(elif_clause.get_stmts_body())

            if if_stmt.has_else_clause():
                self._process_stmts_body(if_stmt.get_else_clause().get_stmts_body())

        elif compound_stmt.is_while_stmt():
            while_stmt = compound_stmt.get_while_stmt()
            while_stmt.condition, _ = self._process_expression(while_stmt.get_condition())
            self._process_stmts_body(while_stmt.get_stmts_body())

        elif compound_stmt.is_for_stmt():
            for_stmt = compound_stmt.get_for_stmt()
            loop_var = ASTNodeFactory.create_ast_variable(for_stmt.get_variable())
            loop_var.update_scope(for_stmt.get_scope())
            target_unit = self._get_target_unit_of_variable(loop_var)
            for_stmt.start_from = self._process_and_convert(for_stmt.get_start_from(), target_unit)
            for_stmt.end_at = self._process_and_convert(for_stmt.get_end_at(), target_unit)
            self._process_stmts_body(for_stmt.get_stmts_body())

    def _process_assignment(self, assignment: ASTAssignment) -> None:
        lhs = assignment.get_variable()
        if lhs.has_vector_parameter():
            lhs.vector_parameter = self._process_and_convert(lhs.get_vector_parameter(), u.dimensionless_unscaled)

        if assignment.is_compound_product or assignment.is_compound_quotient or assignment.is_compound_integer_quotient:
            # the right-hand side should be dimensionless
            assignment.rhs = self._process_and_convert(assignment.get_expression(), u.dimensionless_unscaled)
            return

        target_unit = self._get_target_unit_of_variable(lhs)
        assignment.rhs = self._process_and_convert(assignment.get_expression(), target_unit)


def get_default_unit_system_path(target_platform: str) -> str:
    r"""Return the path to the JSON file that contains the default unit system for the given target platform."""
    resources_dir_for_target = {"NEST": "resources_nest",
                                "NEST_COMPARTMENTAL": "resources_nest_compartmental",
                                "NEST_DESKTOP": "resources_nest_desktop",
                                "NEST_GPU": "resources_nest_gpu",
                                "PYTHON_STANDALONE": "resources_python_standalone",
                                "SPINNAKER": "resources_spinnaker",
                                "GENN": "resources_genn"}

    codegeneration_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "codegeneration")
    return os.path.join(codegeneration_dir, resources_dir_for_target[target_platform.upper()], "unit_system.json")
