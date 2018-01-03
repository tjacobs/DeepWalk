#!/bin/bash

rm CMakeCache.txt
mkdir build
cd build
cmake ..
make -j4

