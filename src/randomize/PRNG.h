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

#ifndef _randomize_PRNG_h
#define _randomize_PRNG_h

#include <ctime>

#include <core_library/UO.h>

#include "RNG.h"

namespace randomize
{
    template < typename D >
    class PRNG : public RNG< D >
    {
    public:
	PRNG( curandRngType_t rng_type, unsigned long long seed = 0 ) : RNG< D >( rng_type ), _seed( seed )
	{
	    reseed( _seed ? _seed : time(NULL) );
	}

	void reseed( unsigned long long seed )
	{
	    _seed = seed;
	    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(this->_gen, _seed));
	}

    private:
	unsigned long long _seed;
    };
}

#endif // !_randomize_PRNG_h
