cmake_minimum_required(VERSION 3.8)
project(Cognitio)


set(CMAKE_CXX_STANDARD 11)


set(SOURCE_FILES
        src/main.cpp
        src/Network.cpp src/Network.h
        src/Utility.h src/MNIST.cpp src/MNIST.h)

add_executable(Cognitio ${SOURCE_FILES})



