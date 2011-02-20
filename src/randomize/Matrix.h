// -*- mode: c++; c-indent-level: 4; c++-member-init-indent: 8; comment-column: 35; -*-

/* This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2
 * as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * Authors: Caner Candan <caner@candan.fr>, http://caner.candan.fr
 */

#ifndef _randomize_Matrix_h
#define _randomize_Matrix_h

#include <cstdlib>
#include <stdexcept>

#include <cuda.h>

namespace randomize
{
    template < typename Atom >
    class Matrix
    {
    public:
	Matrix() : _deviceMatrix(NULL), _n(0), _m(0) {}

	Matrix(int n) : _deviceMatrix(NULL), _n(n), _m(n)
	{
	    createDeviceMatrix(_deviceMatrix, _n, _m);
	}

	Matrix(int n, int m) : _deviceMatrix(NULL), _n(n), _m(m)
	{
	    createDeviceMatrix(_deviceMatrix, _n, _m);
	}

	Matrix(int n, int m, Atom value) : _deviceMatrix(NULL), _n(n), _m(m)
	{
	    Atom* hostMatrix;
	    createHostMatrix(hostMatrix, _n, _m);
	    fillHostMatrix(hostMatrix, _n, _m, value);
	    createDeviceMatrix(_deviceMatrix, _n, _m);
	    memcpyHostToDevice(hostMatrix, _deviceMatrix, _n, _m);
	    destroyHostMatrix(hostMatrix);
	}

	~Matrix()
	{
	    destroyDeviceMatrix(_deviceMatrix);
	}

	Matrix& operator=( Atom*& m )
	{
	    memcpyHostToDevice(m, _deviceMatrix, _n, _m);
	    return *this;
	}

	std::string className() const { return "Matrix"; }

	virtual void printOn(std::ostream& _os) const
	{
	    if ( !_deviceMatrix ) { return; }
	    if ( _n <= 0 ) { return; }
	    if ( _m <= 0 ) { return; }

	    Atom* hostMatrix;
	    createHostMatrix( hostMatrix, _n, _m );

	    CUDA_CALL( cudaMemcpy(hostMatrix, _deviceMatrix, _n*sizeof(*hostMatrix), cudaMemcpyDeviceToHost) );

	    for ( int i = 0; i < _n; ++i )
		{
		    _os << "[" << *(hostMatrix + i*_n);
		    for ( int j = 0; j < _m; ++j )
			{
			    _os << ", " << *(hostMatrix + i*_n + j);
			}
		    _os << "]" << std::endl;;
		}

	    destroyHostMatrix(hostMatrix);
	}

	operator Atom*() const
	{
	    if ( !_deviceMatrix )
		{
		    throw std::runtime_error("deviceMatrix is not allocated on GPU memory");
		}
	    return _deviceMatrix;
	}

	inline int rows() const { return _n; }
	inline int cols() const { return _m; }

	void resize(int n, int m)
	{
	    if ( _deviceMatrix )
		{
		    destroyDeviceMatrix( _deviceMatrix );
		}

	    _n = n;
	    _m = m;
	    createDeviceMatrix(_deviceMatrix, _n, _m);
	}

    private:
	Atom* _deviceMatrix;
	int _n;
	int _m;

    public:
	/// Here's some high level cublas routines in static

	static void createDeviceMatrix(Atom*& deviceMatrix, int n, int m)
	{
	    CUDA_CALL( cudaMalloc(n*m, sizeof(*deviceMatrix), (void**)&deviceMatrix) );
	}

	static void destroyDeviceMatrix(Atom*& deviceMatrix)
	{
	    CUDA_CALL( cublasFree(deviceMatrix) );
	    deviceMatrix = NULL;
	}

	static void createHostMatrix(Atom*& hostMatrix, int n, int m)
	{
	    hostMatrix = (Atom*)malloc(n*m*sizeof(*hostMatrix));
	    if ( hostMatrix == NULL )
		{
		    throw std::runtime_error("out of memory");
		}
	}

	static void destroyHostMatrix(Atom*& hostMatrix)
	{
	    free(hostMatrix);
	    hostMatrix = NULL;
	}

	static void fillHostMatrix(Atom*& hostMatrix, int n, int m, Atom value)
	{
	    for (int j = 0; j < n; ++j)
		{
		    for (int i = 0; i < m; ++i)
			{
			    *(hostMatrix + i + j) = value;
			}
		}
	}

	static void memcpyHostToDevice(Atom*& hostMatrix, Atom*& deviceMatrix, int n, int m)
	{
	    CUDA_CALL( cudaMemcpy(deviceMatrix, hostMatrix, n*m*sizeof(*deviceMatrix), cudaMemcpyHostToDevice) );
	}

	static void memcpyDeviceToHost(Atom*& deviceMatrix, Atom*& hostMatrix, int n, int m)
	{
	    CUDA_CALL( cudaMemcpy(hostMatrix, deviceMatrix, n*m*sizeof(*hostMatrix), cudaMemcpyDeviceToHost) );
	}
    };
}

#endif // !_randomize_Matrix_h
