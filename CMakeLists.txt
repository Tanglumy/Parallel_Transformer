cmake_minimum_required(VERSION 3.17)
project(csi596_project)

set(CMAKE_CXX_STANDARD 14)

add_executable(csi596_project
        baseline/process_seq.py
        baseline/shakespeare.txt
        baseline/transformer.cpp
        baseline/transformer.py
        dataset_vectors.txt
        LICENSE
        README.md
        test.cu)
