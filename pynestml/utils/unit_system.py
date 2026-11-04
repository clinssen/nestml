# -*- coding: utf-8 -*-
#
# unit_system.py
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

from typing import List, Mapping, Optional, Tuple, Union

import json
import math
import os

from astropy import units as u
from astropy.units.physical import PhysicalType

from pynestml.symbols.predefined_units import PredefinedUnits


class UnitSystem:
    r"""
    A unit system determines, for each physical quantity (such as "time" or "electrical potential"), in which unit numerical values of that quantity are expressed in the generated code. For instance, NEST Simulator expresses time in milliseconds and voltages in millivolts, whereas a unit system based on SI units would express them in seconds and volts, respectively.

    The unit system is specified as a mapping from quantity names to unit names, for instance:

    .. code-block:: json

       {
           "time": "ms",
           "electrical potential": "mV",
           "electrical current": "pA"
       }

    The quantity names are those known to astropy (see https://docs.astropy.org/en/stable/units/physical_types.html).

    Quantities that do not appear in the mapping are expressed by combining the units of the quantities that make up their definition. For example, if the mapping contains ``"electrical potential": "mV"`` and ``"time": "ms"``, but no entry for the derived quantity "electrical potential / time", then a variable declared as ``"electrical potential" / time`` is expressed in mV/ms. If, ultimately, no entry is found, the SI unit is used.
    """

    _entries: List[Tuple[PhysicalType, u.UnitBase]]

    def __init__(self, quantity_to_unit: Mapping[str, str], name: str = "custom"):
        r"""
        :param quantity_to_unit: mapping from physical quantity names to unit names, e.g. ``{"time": "ms"}``
        :param name: a name to identify this unit system in messages
        """
        self.name = name
        self._quantity_to_unit = dict(quantity_to_unit)
        self._entries = []
        for quantity_name, unit_name in quantity_to_unit.items():
            physical_type = PredefinedUnits.get_quantity(quantity_name)
            if physical_type is None:
                raise UnitSystemException("Unit system \"" + name + "\": unknown physical quantity \"" + str(quantity_name) + "\"")

            try:
                unit = u.Unit(unit_name)
            except ValueError as e:
                raise UnitSystemException("Unit system \"" + name + "\": cannot interpret unit \"" + str(unit_name) + "\" for quantity \"" + str(quantity_name) + "\"") from e

            if not unit.physical_type == physical_type:
                raise UnitSystemException("Unit system \"" + name + "\": unit \"" + str(unit_name) + "\" (" + str(unit.physical_type) + ") is not a unit of quantity \"" + str(quantity_name) + "\"")

            self._entries.append((physical_type, unit))

    @classmethod
    def from_json_file(cls, path: str) -> UnitSystem:
        r"""
        Load a unit system from a JSON file containing a single object, which maps quantity names to unit names.
        """
        with open(path, "r") as f:
            quantity_to_unit = json.load(f)

        if not isinstance(quantity_to_unit, dict):
            raise UnitSystemException("Unit system file \"" + str(path) + "\" should contain a single JSON object that maps quantity names to unit names")

        return cls(quantity_to_unit, name=os.path.basename(path))

    @classmethod
    def from_option(cls, value: Union[str, Mapping[str, str], UnitSystem]) -> UnitSystem:
        r"""
        Construct a unit system from either the path to a JSON file, or a dictionary mapping quantity names to unit names.
        """
        if isinstance(value, UnitSystem):
            return value

        if isinstance(value, str):
            return cls.from_json_file(value)

        if isinstance(value, Mapping):
            return cls(value)

        raise UnitSystemException("Unit system should be specified as a path to a JSON file, or as a dictionary; got " + str(type(value)))

    def to_dict(self) -> Mapping[str, str]:
        r"""
        Return the mapping from quantity names to unit names that defines this unit system. This can be passed, for instance, as the ``unit_system`` option of the NonDimensionalisationTransformer.
        """
        return dict(self._quantity_to_unit)

    def get_unit_for_quantity(self, physical_type: PhysicalType) -> Optional[u.UnitBase]:
        r"""
        Return the unit specified for the given quantity, or None if the quantity is not part of the mapping.
        """
        for _physical_type, unit in self._entries:
            if _physical_type == physical_type:
                return unit

        return None

    def get_time_unit(self) -> u.UnitBase:
        r"""
        Return the unit in which time is expressed (the unit of the simulation timestep, ``resolution()``, ``t``, etc.).
        """
        return self.get_target_unit(u.s)

    def get_target_unit(self, unit: u.UnitBase) -> u.UnitBase:
        r"""
        For a given unit (for instance, "V"), return the corresponding unit in this unit system, in which numerical values with that unit are expressed (for instance, "mV").

        The returned unit always has a scale of 1 (e.g. "mV" rather than "1000 uV").
        """
        unit = u.Unit(unit)

        if unit.physical_type == "dimensionless":
            return u.dimensionless_unscaled

        target_unit = self.get_unit_for_quantity(unit.physical_type)
        if target_unit is not None:
            return target_unit

        bases = unit.bases
        powers = unit.powers

        if len(bases) == 1 and powers[0] == 1:
            # ``unit`` is a single named unit (for instance, "Hz" or "kg") whose quantity is not in the mapping; decompose it into SI base units (for instance, "1 / s")
            decomposed = unit.decompose()
            if len(decomposed.bases) == 1 and decomposed.powers[0] == 1:
                # this is an SI base unit (or a scaled version of it, for instance "g" for "kg") whose quantity is not in the mapping: use the SI base unit
                return decomposed.bases[0]

            bases = decomposed.bases
            powers = decomposed.powers

        # combine the units of the constituent parts, e.g. "mV / ms" for "V / s"
        target_unit = u.dimensionless_unscaled
        for base, power in zip(bases, powers):
            target_unit = target_unit * self.get_target_unit(base) ** power

        if not unit.physical_type == "unknown":
            # if the combination is equal to the (named) SI unit of the quantity, prefer the named unit, e.g. "V" rather than "m2 kg / (A s3)"
            si_unit = unit.physical_type._unit
            if UnitSystem.is_unity(UnitSystem.get_conversion_factor(target_unit, si_unit)):
                return si_unit

        return target_unit

    @staticmethod
    def get_conversion_factor(from_unit: u.UnitBase, to_unit: u.UnitBase) -> Optional[float]:
        r"""
        Return the factor ``f`` such that a value ``x`` expressed in ``from_unit`` is equal to ``f * x`` expressed in ``to_unit``. Returns None if the two units have different physical dimensions.
        """
        try:
            factor = u.Unit(from_unit).to(u.Unit(to_unit))
        except u.UnitConversionError:
            return None

        return UnitSystem.round_factor(factor)

    @staticmethod
    def round_factor(factor: float) -> float:
        r"""
        Round a conversion factor to 15 significant digits, to get rid of floating-point noise (e.g. 0.0010000000000000002 becomes 0.001).
        """
        if factor == 0 or not math.isfinite(factor):
            return factor

        return float("%.15g" % factor)

    @staticmethod
    def is_unity(factor: float) -> bool:
        return abs(factor - 1.) < 1E-12

    def __str__(self) -> str:
        return "UnitSystem(" + self.name + ": " + ", ".join([str(pt) + " -> " + str(unit) for pt, unit in self._entries]) + ")"


class UnitSystemException(Exception):
    pass
