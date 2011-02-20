// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors:
 * Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

#ifndef _randomize_kernel_boxmuller_h
#define _randomize_kernel_boxmuller_h

#undef PI
#define PI 3.14159265358979f

#undef RNG_COUNT
#define RNG_COUNT 4096

namespace randomize
{
    namespace kernel
    {
	namespace boxmuller
	{
	    template < typename Atom >
	    __device__ inline void BoxMuller( Atom& u1, Atom& u2 )
	    {
		Atom r = sqrtf(-2.0f * logf(u1));
		Atom phi = 2 * PI * u2;
		u1 = r * __cosf(phi);
		u2 = r * __sinf(phi);
	    }

	    template < typename Atom >
	    __global__ void kernel( Atom *data, int size )
	    {
		const int tid = blockDim.x * blockIdx.x + threadIdx.x;
		for ( int i = 0; i < size; i += 2 )
		    {
			BoxMuller(data[tid + (i + 0) * RNG_COUNT],
				  data[tid + (i + 1) * RNG_COUNT]);
		    }
	    }
	}
    }
}

#endif // !_randomize_kernel_boxmuller_h
