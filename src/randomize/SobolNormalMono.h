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

#ifndef _randomize_SobolNormalMono_h
#define _randomize_SobolNormalMono_h

#include "Sobol.h"
#include "NormalMono.h"

namespace randomize
{
    template < typename Atom >
    class SobolNormalMono : public Sobol< NormalMono< Atom > >
    {
    public:
	void operator()( const NormalMono< Atom >& distrib, Array< Atom >& data );
    };

    template <>
    void SobolNormalMono< float >::operator()( const NormalMono< float >& distrib, Array< float >& data )
    {
	CURAND_CALL( curandGenerateNormal(this->_gen, data, data.size(), distrib.mean(), distrib.variance()) );
    }

    template <>
    void SobolNormalMono< double >::operator()( const NormalMono< double >& distrib, Array< double >& data )
    {
	CURAND_CALL( curandGenerateNormalDouble(this->_gen, data, data.size(), distrib.mean(), distrib.variance()) );
    }
}

#endif // !_randomize_SobolNormalMono_h
