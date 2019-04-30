# VehicleDC
# Requirement
Python 3 with pytorch numpy opencv pillow

NVIDIA GPU
# Install
``` bash
git clone https://github.com/LitingLin/VehicleDC.git --recursive --depth 1
cd VehicleDC
mkdir build
cd build
cmake ..
make -j
make install
```
Type ``` make test ``` to check if it is working properly. (libjpeg-turbo is required)
# Usage
In C/C++ source code
``` C++
#include <vehicle_dc/vehicle_dc.h>
```
In Makefile
``` Bash
CFLAGS=-I/usr/local/include
CXXFLAGS=-I/usr/local/include
LDFLAGS=-L/usr/local/lib -Wl,-rpath=/usr/local/lib -lVehicleDC
```
You can find an example in ```test/test.cpp```.
