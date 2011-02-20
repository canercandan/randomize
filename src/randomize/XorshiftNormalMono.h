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

#ifndef _randomize_XorshiftNormalMono_h
#define _randomize_XorshiftNormalMono_h

#include "Xorshift.h"
#include "NormalMono.h"

namespace randomize
{
    template < typename Atom >
    class XorshiftNormalMono : public Xorshift< NormalMono< Atom > >
    {
    public:
	XorshiftNormalMono( unsigned long long seed = 0 ) : Xorshift< NormalMono< Atom > >( seed ) {}

	void operator()( const NormalMono< Atom >& distrib, Data< Atom >& data );
    };

    template <>
    void XorshiftNormalMono< float >::operator()( const NormalMono< float >& distrib, Data< float >& data )
    {
	CURAND_CALL( curandGenerateNormal(this->_gen, data, data.size(), distrib.mean(), distrib.variance()) );
    }

    template <>
    void XorshiftNormalMono< double >::operator()( const NormalMono< double >& distrib, Data< double >& data )
    {
	CURAND_CALL( curandGenerateNormalDouble(this->_gen, data, data.size(), distrib.mean(), distrib.variance()) );
    }
}

#endif // !_randomize_XorshiftNormalMono_h
