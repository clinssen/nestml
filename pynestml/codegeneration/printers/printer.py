# -*- coding: utf-8 -*-
#
# printer.py
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

from pynestml.codegeneration.printers.reference_converter import ReferenceConverter
from pynestml.codegeneration.printers.types_printer import TypesPrinter


class Printer:
    r"""
    By using a different ReferenceConverter and TypesPrinter for the handling of variables, names, and functions and so on, Printers can be easily adapted to different targets.
    """

    def __init__(self, reference_converter: ReferenceConverter, types_printer: TypesPrinter):
        assert isinstance(reference_converter, ReferenceConverter)
        self.reference_converter = reference_converter
        self.types_printer = types_printer
