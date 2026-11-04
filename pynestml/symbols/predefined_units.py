# -*- coding: utf-8 -*-
#
# predefined_units.py
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

from typing import Mapping, Optional, Sequence

from astropy import units as u
from astropy.units.physical import PhysicalType

from pynestml.utils.logger import Logger, LoggingLevel
from pynestml.utils.messages import Messages
from pynestml.utils.unit_type import UnitType


class PredefinedUnits:
    """
    This class represents a collection of physical units. Units can be retrieved by means of get_unit(name).
    Attribute:
        name2unit (dict):  Dict of all predefined units, map from name to unit object.
    """
    name2unit = {}   # type: Mapping[str, UnitType]

    @classmethod
    def register_units(cls):
        """
        Registers all units in astropy.units (more specifically, from the si, cgs and astrophys submodules) as predefined units into NESTML.
        """
        # first store all base units and the derived units without the prefix in a list
        cls.name2unit = {}

        for unit_str in dir(u.si) + dir(u.cgs) + dir(u.astrophys):
            try:
                unit = eval("u." + unit_str)    # grab the unit object
            except BaseException:
                unit = None

            if issubclass(type(unit), u.core.UnitBase):
                for unit_name in unit.names:
                    if unit_name == "min":
                        # this conflicts with the predefined function ``min()``
                        continue

                    temp_unit = UnitType(name=str(unit_name), unit=unit)
                    cls.name2unit[str(unit_name)] = temp_unit

    @classmethod
    def get_unit(cls, name: str) -> UnitType:
        """
        Returns a single UnitType if the corresponding unit has been predefined.
        :param name: the name of a unit
        :return: a single UnitType object, or None
        """
        if name in cls.name2unit.keys():
            return cls.name2unit[name]
        else:
            code, message = Messages.get_unit_does_not_exist(name)
            Logger.log_message(code=code, message=message, log_level=LoggingLevel.ERROR)
            return None

    @classmethod
    def is_unit(cls, name: str) -> bool:
        """
        Indicates whether the handed over name represents a stored unit.
        :param name: a single name
        :return: True if unit name, otherwise False.
        """
        return name in cls.name2unit.keys()

    @classmethod
    def register_unit(cls, unit: UnitType) -> None:
        """
        Registers the handed over unit in the set of the predefined units.
        :param unit: a single unit type.
        """
        if unit.get_name() is not cls.name2unit.keys():
            cls.name2unit[unit.get_name()] = unit

    @classmethod
    def get_quantity(cls, name: str) -> Optional[PhysicalType]:
        r"""
        Returns the physical quantity (as an astropy ``PhysicalType``) with the given name, for instance, ``"time"`` or ``"electrical potential"``, or None if no such quantity is known. The list of known quantity names is defined by astropy (see https://docs.astropy.org/en/stable/units/physical_types.html).

        :param name: the name of a physical quantity
        :return: the corresponding physical type, or None
        """
        try:
            physical_type = u.get_physical_type(name)
        except (ValueError, TypeError, KeyError):
            return None

        if str(physical_type) == "unknown":
            return None

        return physical_type

    @classmethod
    def is_quantity(cls, name: str) -> bool:
        r"""
        Indicates whether the handed over name represents a physical quantity, for instance, ``"time"`` or ``"electrical potential"``.

        :param name: a single name
        :return: True if quantity name, otherwise False.
        """
        return cls.get_quantity(name) is not None

    @classmethod
    def get_quantity_si_unit(cls, name: str) -> Optional[u.UnitBase]:
        r"""
        Returns the SI unit corresponding to a physical quantity, for instance, ``s`` for ``"time"`` and ``V`` for ``"electrical potential"``, or None if no quantity by this name is known.

        :param name: the name of a physical quantity
        :return: the SI unit of the quantity, or None
        """
        physical_type = cls.get_quantity(name)
        if physical_type is None:
            return None

        return physical_type._unit    # astropy does not provide a public API to obtain the SI unit of a ``PhysicalType``

    @classmethod
    def get_units(cls) -> Sequence[UnitType]:
        """
        Returns the list of all currently defined units.
        :return: a list of all defined units.
        """
        return cls.name2unit
