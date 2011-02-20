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

#ifndef _randomize_NormalMono_h
#define _randomize_NormalMono_h

#include "Distrib.h"

namespace randomize
{
    template < typename Atom >
    class NormalMono : public Distrib< Atom >
    {
    public:
	NormalMono( const Atom& mean, const Atom& variance )
	    : _mean(mean), _variance(variance)
	{
	    assert(_mean.size() > 0);
	    assert(_mean.size() == _variance.size());
	}

	unsigned int size()
	{
	    assert(_mean.size() == _variance.size());
	    return _mean.size();
	}

	Atom mean(){return _mean;}
	Atom variance(){return _variance;}

    private:
	Atom _mean;
	Atom _variance;
    };
}

#endif // !_randomize_NormalMono_h
