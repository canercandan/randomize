/*
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

Copyright (C) 2010 Thales group
*/
/*
Authors:
    Johann Dréo <johann.dreo@thalesgroup.com>
    Caner Candan <caner.candan@thalesgroup.com>
*/

#ifndef _randomize_Uniform_h
#define _randomize_Uniform_h

#include "Distrib.h"

namespace randomize
{
    template < typename Atom >
    class Uniform : public Distrib< Atom >
    {
    public:
	Uniform(Atom min, Atom max) : _min(min), _max(max) {}

	Atom min() const { return _min; }
	Atom max() const { return _max; }

    private:
	Atom _min;
	Atom _max;
    };
}

#endif // !_randomize_Uniform_h
