#!/bin/bash

mkdir -p build/make
cd build/make
cmake ../..
make
./neural_net