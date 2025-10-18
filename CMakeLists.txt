cmake_minimum_required(VERSION 4.0)
project(ROLab2)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)

add_executable(ROLab2 parallel.cpp
serial.cpp)

target_link_libraries(ROLab2 PRIVATE
        MPI::MPI_CXX  # The C++ MPI target
        m             # The math library (for sqrt())
)
