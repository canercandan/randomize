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

#ifndef _randomize_BoxMuller_h
#define _randomize_BoxMuller_h

#include "InverseCumulativeNormal.h"

#include "kernel/boxmuller.h"

namespace randomize
{
    template < typename Atom >
    class BoxMuller : public InverseCumulativeNormal< Atom >
    {
    public:
	BoxMuller( int nblocks = 32, int nthreads = 128 ) : _nblocks(nblocks), _nthreads(nthreads) {}

	void operator()( Array< Atom >& data )
	{
	    //const int size = iAlignUp( iDivUp( data.size(), _nblocks*_nthreads ), 2 );
	    kernel::boxmuller::kernel< Atom ><<<_nblocks, _nthreads>>>( data, data.size() );
	}

    private:
	int _nblocks;
	int _nthreads;

    public:
	//ceil(a / b)
	static inline int iDivUp(int a, int b)
	{
	    return ((a % b) != 0) ? (a / b + 1) : (a / b);
	}

	//Align a to nearest higher multiple of b
	static inline int iAlignUp(int a, int b)
	{
	    return ((a % b) != 0) ?  (a - a % b + b) : a;
	}
    };
}

#endif // !_randomize_BoxMuller_h
