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

#ifndef _randomize_SobolUniform_h
#define _randomize_SobolUniform_h

#include "Sobol.h"
#include "Uniform.h"

namespace randomize
{
    template < typename Atom >
    class SobolUniform : public Sobol< Uniform< Atom > >
    {
    public:
	void operator()( const Uniform< Atom >& distrib, Array< Atom >& data );
    };

    template <>
    void SobolUniform< float >::operator()( const Uniform< float >& distrib, Array< float >& data )
    {
	CURAND_CALL( curandGenerateUniform(this->_gen, data, data.size()) );
    }

    template <>
    void SobolUniform< double >::operator()( const Uniform< double >& distrib, Array< double >& data )
    {
	CURAND_CALL( curandGenerateUniformDouble(this->_gen, data, data.size()) );
    }
}

#endif // !_randomize_SobolUniform_h
