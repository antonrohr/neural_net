#!/bin/bash

mkdir -p build/xcode
cd build/xcode
cmake -G Xcode ../..
open neural_net_cpp.xcodeproj