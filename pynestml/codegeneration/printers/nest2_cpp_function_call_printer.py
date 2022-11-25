# -*- coding: utf-8 -*-
#
# nest2_cpp_function_call_printer.py
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

from pynestml.codegeneration.printers.nest_cpp_function_call_printer import NESTCppFunctionCallPrinter
from pynestml.meta_model.ast_function_call import ASTFunctionCall
from pynestml.symbols.predefined_functions import PredefinedFunctions


class NEST2CppFunctionCallPrinter(NESTCppFunctionCallPrinter):
    r"""
    This class is used to convert operators and constants to the GSL (GNU Scientific Library) processable format.
    """

    def _print_function_call_format_string(self, function_call: ASTFunctionCall, prefix: str = "") -> str:
        r"""Convert a single function call to C++ GSL API syntax.

        Parameters
        ----------
        function_call : ASTFunctionCall
            The function call node to convert.
        prefix : str
            Optional string that will be prefixed to the function call. For example, to refer to a function call in the class "node", use a prefix equal to "node." or "node->".

            Predefined functions will not be prefixed.

        Returns
        -------
        s : str
            The function call string in C++ syntax.
        """
        function_name = function_call.get_name()

        if function_name == PredefinedFunctions.RANDOM_NORMAL:
            return '(({!s}) + ({!s}) * ' + prefix + 'normal_dev_( nest::kernel().rng_manager.get_rng( ' + prefix + 'get_thread() ) ))'

        if function_name == PredefinedFunctions.RANDOM_UNIFORM:
            return '(({!s}) + ({!s}) * nest::kernel().rng_manager.get_rng( ' + prefix + 'get_thread() )->drand())'

        return super()._print_function_call_format_string(function_call, prefix=prefix)
