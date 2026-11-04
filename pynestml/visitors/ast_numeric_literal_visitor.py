# -*- coding: utf-8 -*-
#
# ast_numeric_literal_visitor.py
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

from pynestml.meta_model.ast_node import ASTNode
from pynestml.meta_model.ast_unit_type import ASTUnitType
from pynestml.symbols.predefined_types import PredefinedTypes
from pynestml.symbols.predefined_units import PredefinedUnits
from pynestml.symbols.symbol import SymbolKind
from pynestml.utils.logger import Logger, LoggingLevel
from pynestml.utils.messages import Messages
from pynestml.visitors.ast_data_type_visitor import ASTDataTypeVisitor
from pynestml.visitors.ast_visitor import ASTVisitor


class ASTNumericLiteralVisitor(ASTVisitor):
    """
    Visits a single numeric literal and updates its type.
    """

    def _check_no_quantities_in_unit_type(self, node: ASTNode, unit_type: ASTUnitType) -> None:
        r"""Physical quantities (such as "time") can only be used as data types, not as the unit of a numeric literal: log an error if a quantity is used."""
        if unit_type.is_simple_unit():
            is_variable = node.get_scope() is not None and node.get_scope().resolve_to_symbol(unit_type.unit, SymbolKind.VARIABLE) is not None    # variables that appear inside the unit type will be split off by the UnitTypeFixerVisitor
            if unit_type.is_quoted or (not is_variable and not PredefinedUnits.is_unit(unit_type.unit) and PredefinedUnits.is_quantity(unit_type.unit)):
                code, message = Messages.get_quantity_used_as_unit_of_literal(str(node), unit_type.unit)
                Logger.log_message(code=code, message=message, error_position=node.get_source_position(), log_level=LoggingLevel.ERROR)

            return

        for child in unit_type.get_children():
            if isinstance(child, ASTUnitType):
                self._check_no_quantities_in_unit_type(node, child)

    def visit_simple_expression(self, node: ASTNode) -> None:
        """
        Visit a simple rhs and update the type of a numeric literal.
        :param node: a single meta_model node
        :return: no value returned, the type is updated in-place
        """
        assert node.get_scope() is not None, "Run symboltable creator."
        # if unitType is set in this rhs, the var type overrides the literal
        if node.unitType is not None:
            self._check_no_quantities_in_unit_type(node, node.unitType)
            node.unitType.accept(ASTDataTypeVisitor())
            node.type = node.unitType.get_type_symbol()
            node.type.referenced_object = node
            return

        if node.get_numeric_literal() is not None and isinstance(node.get_numeric_literal(), float):
            node.type = PredefinedTypes.get_real_type()
            node.type.referenced_object = node
            return

        elif node.get_numeric_literal() is not None and isinstance(node.get_numeric_literal(), int):
            node.type = PredefinedTypes.get_integer_type()
            node.type.referenced_object = node
            return
