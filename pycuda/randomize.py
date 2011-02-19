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

parser = optparse.OptionParser()
parser.add_option('-p', '--plot', action='store_true', default=False, help='plot data')
parser.add_option('-r', '--nruns', type='int', default=1, help='give here a number of runs')
parser.add_option('-b', '--nblocks', type='int', default=128, help='give here a number of blocks to use')
parser.add_option('-t', '--nthreads', type='int', default=64, help='give here a number of threads to use')
options = common.parser(parser)

if options.plot:
    import pylab

def justtest():
    nbr_values = options.nblocks * options.nthreads

    logger.info("Parameters:")
    logger.info("- nbr_values = %d" % nbr_values)
    logger.info("- nruns = %d" % options.nruns)

    start = drv.Event()
    end = drv.Event()

    """
    Elementwise SECTION
    use an ElementwiseKernel with sin in a
    for loop all in C call from Python
    """
    kernel = ElementwiseKernel(
        "float* data, int nruns",
        """
        for (int r = 0; r < nruns; ++r)
        {
            data[i] = erfc(data[i]);
        }
        """,
        "gpuerfc")

    mtkernel = ElementwiseKernel(
        "float* data, int nPerRng",
        """
        int iState, iState1, iStateM, iOut;
        unsigned int mti, mti1, mtiM, x;
        unsigned int mt[MT_NN], matrix_a, mask_b, mask_c;
        """,
        "mtrand")

    data = gpuarray.to_gpu( numpy.ones(nbr_values).astype(numpy.float32) )

    start.record()
    kernel( data, options.nruns )
    end.record()

    end.synchronize()
    secs = start.time_till(end)*1e-3

    hostdata = data.get()

    logger.info("Elementwise time and first three results:")
    logger.info("%fs, %s, %d" % (secs, str(hostdata[:3]), len(hostdata)))

def main():
    justtest()

if __name__ == '__main__':
    main()
