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

import numpy

import pycuda.driver as drv
# import pycuda.tools
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# import pycuda.cumath

from pycuda.elementwise import ElementwiseKernel
# from pycuda.compiler import SourceModule

import optparse, logging
import common

logger = logging.getLogger("Randomizer")

logger.info("TEST")

options = common.parser()

def justtest():
    blocks = 64
    block_size = 128
    nbr_values = blocks * block_size

    logger.info("Using nbr_values == %d" % nbr_values)

    n_iter = 100000

    logger.info("Calculating %d iterations" % n_iter)

    start = drv.Event()
    end = drv.Event()

    """
    Elementwise SECTION
    use an ElementwiseKernel with sin in a
    for loop all in C call from Python
    """
    kernel = ElementwiseKernel(
        "float *data, int n_iter",
        #"for(int n = 0; n < n_iter; n++) { data[i] = sin(data[i]);}",
        #"for(int n = 0; n < n_iter; n++) { data[i] = i;}",
        "data[i] = erfc(data[i]);",
        "gpuerfc")

    data = gpuarray.to_gpu( numpy.ones(nbr_values).astype(numpy.float32) )

    start.record()
    kernel( data, numpy.int(n_iter) )
    end.record()

    end.synchronize()
    secs = start.time_till(end)*1e-3

    logger.info("Elementwise time and first three results:")
    logger.info("%fs, %s" % (secs, str(data.get()[:3])))

def main():
    justtest()

if __name__ == '__main__':
    main()
