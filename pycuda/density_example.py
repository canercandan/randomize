#!/usr/bin/env python

#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2
# as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
# Authors:
# Caner Candan <caner@candan.fr>, http://caner.candan.fr
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps

np.random.seed(0)

shape, scale = 0., 1.
#x = np.random.mtrand.normal(shape, scale, 100)
x = np.random.randn(100000000)
#x = [12, 13, 2]
count, bins, ignored = plt.hist(x, 30, normed=True, cumulative=0)
y = bins**(shape)*((np.exp(-bins/scale))/(sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')
plt.savefig( 'density.pdf', format='pdf' )
plt.cla()
plt.clf()
