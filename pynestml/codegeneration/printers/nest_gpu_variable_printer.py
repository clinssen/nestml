# -*- coding: utf-8 -*-
#
# nest_variable_printer.py
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
from pynestml.codegeneration.nest_gpu_code_generator_utils import NESTGPUCodeGeneratorUtils

from pynestml.utils.ast_utils import ASTUtils

from pynestml.codegeneration.printers.cpp_variable_printer import CppVariablePrinter
from pynestml.codegeneration.printers.expression_printer import ExpressionPrinter
from pynestml.codegeneration.nest_unit_converter import NESTUnitConverter
from pynestml.meta_model.ast_variable import ASTVariable
from pynestml.symbols.predefined_units import PredefinedUnits
from pynestml.symbols.predefined_variables import PredefinedVariables
from pynestml.symbols.symbol import SymbolKind
from pynestml.symbols.unit_type_symbol import UnitTypeSymbol
from pynestml.utils.logger import Logger, LoggingLevel
from pynestml.utils.messages import Messages


class NESTGPUVariablePrinter(CppVariablePrinter):
    r"""
    Variable printer for the NEST-GPU API.
    """

    def __init__(self, expression_printer: ExpressionPrinter, with_origin: bool = True, with_vector_parameter: bool = True) -> None:
        super().__init__(expression_printer)
        self.with_origin = with_origin
        self.with_vector_parameter = with_vector_parameter

    def print_variable(self, variable: ASTVariable) -> str:
        """
        Converts a single variable to nest processable format.
        :param variable: a single variable.
        :return: a nest processable format.
        """
        assert isinstance(variable, ASTVariable)

        if variable.get_name() == PredefinedVariables.E_CONSTANT:
            return "M_E"

        symbol = variable.get_scope().resolve_to_symbol(variable.get_complete_name(), SymbolKind.VARIABLE)

        if symbol is None:
            # test if variable name can be resolved to a type
            if PredefinedUnits.is_unit(variable.get_complete_name()):
                return str(NESTUnitConverter.get_factor(PredefinedUnits.get_unit(variable.get_complete_name()).get_unit()))

            code, message = Messages.get_could_not_resolve(variable.get_name())
            Logger.log_message(log_level=LoggingLevel.ERROR, code=code, message=message,
                               error_position=variable.get_source_position())
            return ""

        if symbol.is_inline_expression:
            # there might not be a corresponding defined state variable; insist on calling the getter function
            return "get_" + self._print(variable, symbol, with_origin=False) + "()"

        assert not symbol.is_kernel(), "Cannot print kernel; kernel should have been converted during code generation"

        if symbol.is_state() or symbol.is_inline_expression:
            return self._print(variable, symbol, with_origin=self.with_origin)

        return self._print(variable, symbol, with_origin=self.with_origin)

    def _print(self, variable: ASTVariable, symbol, with_origin: bool = True) -> str:
        variable_name = CppVariablePrinter._print_cpp_name(variable.get_complete_name())

        if symbol.is_local():
            return variable_name
        
        if with_origin:
            return NESTGPUCodeGeneratorUtils.print_symbol_origin(symbol, variable) % ("i_" + variable_name)

        return variable_name
