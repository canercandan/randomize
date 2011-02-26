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

#ifndef _randomize_common_h
#define _randomize_common_h

#include <sstream>

#include <cuda.h>

#define TEST_CALL(x, status)				\
    do							\
	{						\
	    if ( (x) != status )			\
		{					\
		    std::ostringstream ss;		\
		    ss << "Error at " << __FILE__	\
		       << ":" << __LINE__;		\
		    throw std::runtime_error(ss.str());	\
		}					\
	}						\
    while(0)

#define CUDA_CALL(x) TEST_CALL( (x), cudaSuccess )
#define CURAND_CALL(x) TEST_CALL( (x), CURAND_STATUS_SUCCESS )

#endif // !_randomize_common_h
