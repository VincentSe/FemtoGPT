cmake_minimum_required(VERSION 3.14)

# set the project name and version
project(CPP_Testing_Sample VERSION 1.0)

# define C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# add the executable
add_executable(BinaryClassification main.cpp SupportVectorMachine.cpp CategoricalDecisionTree.cpp Regression.cpp
    MonteCarloTreeSearch.cpp CharTransformer.cpp ParamFunc.cpp)

# specific include directories
target_include_directories(BinaryClassification PUBLIC
                           "${PROJECT_BINARY_DIR}"
                           )

# create compile commands for intelli sense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
