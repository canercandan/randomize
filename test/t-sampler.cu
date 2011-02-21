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

#include <randomize/randomize>

using namespace core_library;
using namespace randomize;

typedef float Atom;
//typedef double Atom;

int main(void)
{
    NormalMono< Atom > distribution( 10, 10 );
    //Uniform< Atom > distribution( 10, 10 );

    //MTNormalMono< Atom > rng;
    //MTUniform< Atom > rng;
    XorshiftNormalMono< Atom > rng( 42 );
    //XorshiftUniform< Atom > rng;
    //SobolNormalMono< Atom > rng;
    //SobolUniform< Atom > rng;

    SamplerNormalMono< Atom > sampler( rng );
    //SamplerUniform< Atom > sampler( rng );

    /// Only for normal distribution samples
    //ErfInv< Atom > inverse;
    BoxMuller< Atom > inverse;
    //Marsaglia< Atom > inverse;

    //Vector< Atom > sample;
    Vector< Atom > sample( 1 << 2 );
    //Matrix< Atom > sample( 100, 100 );
    //Data< Atom > sample;

    // logger << sample << std::endl;
    sampler( distribution, sample );
    logger << sample << std::endl;
    inverse( sample );
    logger << sample << std::endl;

    return 0;
}
