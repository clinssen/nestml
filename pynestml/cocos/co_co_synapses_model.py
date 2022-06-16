# -*- coding: utf-8 -*-
#
# co_co_synapses_model.py
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
from pynestml.meta_model.ast_neuron import ASTNeuron
from pynestml.utils.syns_processing import SynsProcessing


class CoCoSynapsesModel(CoCo):
    
    
    @classmethod
    def check_co_co(cls, neuron: ASTNeuron):
        """
        Checks if this compartmental conditions apply for the handed over neuron. 
        If yes, it checks the presence of expected functions and declarations.
        :param neuron: a single neuron instance.
        :type neuron: ast_neuron
        """
        return SynsProcessing.check_co_co(neuron)

