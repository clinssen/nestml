# -*- coding: utf-8 -*-
#
# debug_data_type_printer.py
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

from pynestml.codegeneration.printers.ast_printer import ASTPrinter
from pynestml.meta_model.ast_node import ASTNode
from pynestml.symbols.type_symbol import TypeSymbol
from pynestml.symbols.real_type_symbol import RealTypeSymbol
from pynestml.symbols.boolean_type_symbol import BooleanTypeSymbol
from pynestml.symbols.integer_type_symbol import IntegerTypeSymbol
from pynestml.symbols.string_type_symbol import StringTypeSymbol
from pynestml.symbols.void_type_symbol import VoidTypeSymbol
from pynestml.symbols.unit_type_symbol import UnitTypeSymbol
from pynestml.symbols.error_type_symbol import ErrorTypeSymbol
from pynestml.utils.either import Either


class DebugDataTypePrinter(ASTPrinter):
    """
    Returns a string format that is suitable for info/warning/error messages.
    """

    def _print_type_symbol(self, type_symbol: TypeSymbol, prefix: str = "") -> str:
        if 'is_buffer' in dir(type_symbol) and type_symbol.is_buffer:
            return 'buffer'

        if isinstance(type_symbol, RealTypeSymbol):
            return 'real'

        if isinstance(type_symbol, BooleanTypeSymbol):
            return 'bool'

        if isinstance(type_symbol, IntegerTypeSymbol):
            return 'int'

        if isinstance(type_symbol, StringTypeSymbol):
            return 'str'

        if isinstance(type_symbol, VoidTypeSymbol):
            return 'void'

        if isinstance(type_symbol, UnitTypeSymbol):
            return type_symbol.unit.unit.to_string()

        if isinstance(type_symbol, ErrorTypeSymbol):
            return '<Error type>'

        return str(type_symbol)

    def print(self, node: ASTNode, prefix: str = "") -> str:
        r"""
        Converts the name of the type symbol to a corresponding nest representation.
        :param type_symbol: a single type symbol
        :return: the corresponding string representation.
        """

        type_symbol = node.get_type_symbol()

        if isinstance(type_symbol, Either):
            if type_symbol.is_value():
                return self._print_type_symbol(type_symbol.get_value())

            assert type_symbol.is_error()
            return type_symbol.get_error()

        return self._print_type_symbol(type_symbol)
