# -*- coding: utf-8 -*-
#
# co_co_all_variables_defined.py
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

from pynestml.cocos.co_co import CoCo
from pynestml.meta_model.ast_declaration import ASTDeclaration
from pynestml.meta_model.ast_model import ASTModel
from pynestml.meta_model.ast_variable import ASTVariable
from pynestml.symbols.symbol import SymbolKind
from pynestml.symbols.variable_symbol import BlockType
from pynestml.utils.logger import Logger, LoggingLevel
from pynestml.utils.messages import Messages
from pynestml.visitors.ast_visitor import ASTVisitor


class CoCoAllVariablesDefined(CoCo):
    """
    This class represents a constraint condition which ensures that all elements as used in expressions have been
    previously defined.
    Not allowed:
        state:
            V_m mV = V_m + 10mV # <- recursive definition
            V_m mV = V_n # <- not defined reference
    """

    @classmethod
    def check_co_co(cls, node: ASTModel, after_ast_rewrite: bool = False):
        """
        Checks if this coco applies for the handed over neuron. Models which contain undefined variables are not correct.
        :param node: a single neuron instance.
        :param after_ast_rewrite: indicates whether this coco is checked after the code generator has done rewriting of the abstract syntax tree. If True, checks are not as rigorous. Use False where possible.
        """
        # for each variable in all expressions, check if the variable has been defined previously
        expression_collector_visitor = ASTExpressionCollectorVisitor()
        node.accept(expression_collector_visitor)
        expressions = expression_collector_visitor.ret
        for expr in expressions:
            if isinstance(expr, ASTVariable):
                vars = [expr]
            else:
                vars = expr.get_variables()

            for var in vars:
                symbol = var.get_scope().resolve_to_symbol(var.get_complete_name(), SymbolKind.VARIABLE)

                # test if the symbol has been defined at least
                if symbol is None:
                    # if after_ast_rewrite:   # after ODE-toolbox transformations, convolutions are replaced by state variables, so cannot perform this check properly
                    #     symbol2 = node.get_scope().resolve_to_symbol(var.get_name(), SymbolKind.VARIABLE)
                    #     if symbol2 is not None:
                    #         # an inline expression defining this variable name (ignoring differential order) exists
                    #         if "__conv__" in str(symbol2):     # if this variable was the result of a convolution...
                    #             continue

                    # check if this symbol is actually a type, e.g. "mV" in the expression "(1 + 2) * mV"
                    symbol2 = var.get_scope().resolve_to_symbol(var.get_complete_name(), SymbolKind.TYPE)
                    if symbol2 is not None:
                        continue  # symbol is a type symbol

                    code, message = Messages.get_variable_not_defined(var.get_complete_name())
                    Logger.log_message(code=code, message=message, error_position=node.get_source_position(),
                                       log_level=LoggingLevel.ERROR, node=node)
                    continue

                # check if it is part of an invariant
                # if it is the case, there is no "recursive" declaration
                # so check if the parent is a declaration and the expression the invariant
                expr_par = expr.get_parent()
                if isinstance(expr_par, ASTDeclaration) and expr_par.get_invariant() == expr:
                    # in this case its ok if it is recursive or defined later on
                    continue

                # check if it has been defined before usage, except for predefined symbols, input ports and variables added by the AST transformation functions
                if (not symbol.is_predefined) \
                        and symbol.block_type != BlockType.INPUT \
                        and not symbol.get_referenced_object().get_source_position().is_added_source_position():
                    # except for parameters, those can be defined after
                    if ((not symbol.get_referenced_object().get_source_position().before(var.get_source_position()))
                            and (not symbol.block_type in [BlockType.PARAMETERS, BlockType.INTERNALS, BlockType.STATE])):
                        code, message = Messages.get_variable_used_before_declaration(var.get_name())
                        Logger.log_message(node=node, message=message, error_position=var.get_source_position(),
                                           code=code, log_level=LoggingLevel.ERROR)
                    # now check that they are not defined recursively, e.g. V_m mV = V_m + 1
                    # todo: we should not check this for invariants
                    if (symbol.get_referenced_object().get_source_position().encloses(var.get_source_position())
                            and not symbol.get_referenced_object().get_source_position().is_added_source_position()):
                        code, message = Messages.get_variable_defined_recursively(var.get_name())
                        Logger.log_message(code=code, message=message, error_position=symbol.get_referenced_object().
                                           get_source_position(), log_level=LoggingLevel.ERROR, node=node)


class ASTExpressionCollectorVisitor(ASTVisitor):

    def __init__(self):
        super(ASTExpressionCollectorVisitor, self).__init__()
        self.ret = list()

    def visit_assignment(self, node):
        self.ret.append(node.get_variable())

    def visit_expression(self, node):
        self.ret.append(node)

    def traverse_expression(self, node):
        # we deactivate traversal in order to get only the root of the expression
        pass

    def visit_simple_expression(self, node):
        self.ret.append(node)

    def traverse_simple_expression(self, node):
        # we deactivate traversal in order to get only the root of the expression
        pass
