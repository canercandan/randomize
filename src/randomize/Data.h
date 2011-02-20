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

#ifndef _randomize_Data_h
#define _randomize_Data_h

#include <cstdlib>
#include <stdexcept>

#include <cuda.h>

#include <core_library/Object.h>
#include <core_library/Logger.h>

#define CUDA_CALL(x)							\
    do									\
	{								\
	    if ( (x) != cudaSuccess )					\
		{							\
		    core_library::logger << core_library::errors	\
					 << "Error at " << __FILE__	\
					 << ":" << __LINE__ << std::endl; \
		    throw std::runtime_error("");			\
		}							\
	}								\
    while(0)

namespace randomize
{
    template < typename Atom >
    class Data : public core_library::Object
    {
    public:
	Data() : _deviceData(NULL), _size(0) {}

	Data(int n) : _deviceData(NULL), _size(n)
	{
	    createDeviceData(_deviceData, _size);
	}

	Data(Atom value) : _deviceData(NULL), _size(1)
	{
	    createDeviceData(_deviceData, _size);
	    memcpyHostToDevice(&value, _deviceData, _size);
	}

	Data(int n, Atom value) : _deviceData(NULL), _size(n)
	{
	    Atom* hostData;
	    createHostData(hostData, _size);
	    fillHostData(hostData, _size, value);
	    createDeviceData(_deviceData, _size);
	    memcpyHostToDevice(hostData, _deviceData, _size);
	    destroyHostData(hostData);
	}

	~Data()
	{
	    if ( !_deviceData ) { return; }
	    destroyDeviceData(_deviceData);
	}

	Data& operator=( Atom*& data )
	{
	    if ( !_deviceData )
		{
		    throw std::runtime_error("deviceData is not allocated on GPU memory");
		}
	    memcpyHostToDevice(data, _deviceData, _size);
	    return *this;
	}

	virtual std::string className() const { return "Data"; }

	virtual void printOn(std::ostream& _os) const
	{
	    if ( !_deviceData ) { return; }
	    if ( _size <= 0 ) { return; }

	    Atom* hostData;
	    createHostData( hostData, _size );

	    CUDA_CALL( cudaMemcpy(hostData, _deviceData, _size*sizeof(*hostData), cudaMemcpyDeviceToHost) );

	    _os << "[" << hostData[0];
	    for ( int i = 1; i < _size; ++i )
		{
		    _os << ", " << hostData[i];
		}
	    _os << "]";

	    destroyHostData(hostData);
	}

	operator Atom*() const
	{
	    if ( !_deviceData )
		{
		    throw std::runtime_error("deviceData is not allocated on GPU memory");
		}
	    return _deviceData;
	}

	inline int size() const { return _size; }

	void resize(int size)
	{
	    if ( _deviceData )
		{
		    destroyDeviceData( _deviceData );
		}

	    _size = size;
	    createDeviceData(_deviceData, _size);
	}

    protected:
	Atom* _deviceData;
	int _size;

    public:
	/// Here's some high level cublas routines in static

	static void createDeviceData(Atom*& deviceData, int n)
	{
	    CUDA_CALL( cudaMalloc((void**)&deviceData, n * sizeof(*deviceData)) );
	}

	static void destroyDeviceData(Atom*& deviceData)
	{
	    CUDA_CALL( cublasFree(deviceData) );
	    deviceData = NULL;
	}

	static void createHostData(Atom*& hostData, int n)
	{
	    hostData = (Atom*)malloc(n*sizeof(*hostData));
	    if ( hostData == NULL )
		{
		    throw std::runtime_error("out of memory");
		}
	}

	static void destroyHostData(Atom*& hostData)
	{
	    free(hostData);
	    hostData = NULL;
	}

	static void fillHostData(Atom*& hostData, int n, Atom value)
	{
	    for (int i = 0; i < n; ++i)
		{
		    *(hostData + i) = value;
		}
	}

	static void memcpyHostToDevice(Atom*& hostData, Atom*& deviceData, int n)
	{
	    CUDA_CALL( cudaMemcpy(deviceData, hostData, n*sizeof(*deviceData), cudaMemcpyHostToDevice) );
	}

	static void memcpyDeviceToHost(Atom*& deviceData, Atom*& hostData, int n)
	{
	    CUDA_CALL( cudaMemcpy(hostData, deviceData, n*sizeof(*hostData), cudaMemcpyDeviceToHost) );
	}

    };
}

#endif // !_randomize_Data_h
