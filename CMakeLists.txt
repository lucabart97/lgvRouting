cmake_minimum_required(VERSION 3.18...3.19)

project(lgvRouting VERSION 1.0
                   DESCRIPTION "lgvRouting project"
                   LANGUAGES CXX)

# set build type
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# suppress preprocessor warning 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLGV_PATH='\"${CMAKE_CURRENT_SOURCE_DIR}/\"'")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -Wno-deprecated-declarations -Wno-unused-variable")

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
enable_language(CUDA)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr)
find_package(CUDAToolkit REQUIRED)

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

#-------------------------------------------------------------------------------
# common library
#-------------------------------------------------------------------------------
add_library(common SHARED 
    src/common/common.cpp
    src/common/log.cpp 
    src/common/CmdParser.cpp
    src/common/Time.cpp
    src/data/Location.cpp
    src/data/Problem.cpp
    src/data/Dataset.cpp
    src/data/Mission.cpp
    src/data/Solution.cpp
)
target_include_directories(common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)
target_link_libraries(common 
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
    CUDA::cudart 
    CUDA::curand 
    CUDA::cuda_driver
    OpenMP::OpenMP_CXX 
)
add_library(${PROJECT_NAME}::common ALIAS common)

add_library(heuristic SHARED 
    src/finder/Finder.cpp
    src/heuristic/Constructive.cpp
    src/heuristic/Generic.cpp
    src/heuristic/LocalSearch.cpp
    src/heuristic/MultiStart.cpp
    src/heuristic/TabuSearch.cpp
    src/heuristic/SimulatedAnnealing.cpp
    src/heuristic/MultiStartGpu.cpp
    src/heuristic/MultiStartMultithread.cpp
)
target_include_directories(heuristic PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)
target_link_libraries(heuristic 
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
    CUDA::cudart 
    CUDA::curand 
    CUDA::cuda_driver
    OpenMP::OpenMP_CXX 
)
add_library(${PROJECT_NAME}::heuristic ALIAS heuristic)

add_library(multistartgpu SHARED 
    src/heuristic/MultiStartGpu.cu
)
target_include_directories(multistartgpu PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)
add_library(${PROJECT_NAME}::multistartgpu ALIAS multistartgpu)


#-------------------------------------------------------------------------------
# Build apps
#-------------------------------------------------------------------------------
file(GLOB lgv_package_APPScpp "app/*.cpp")
foreach(cpp ${lgv_package_APPScpp})
    get_filename_component(app "${cpp}" NAME_WE)
    list(APPEND lgv_package_APPS ${app})
endforeach()
foreach(app ${lgv_package_APPS})
    add_executable(${app} "app/${app}.cpp")
    target_link_libraries(${app} ${PROJECT_NAME}::common ${PROJECT_NAME}::heuristic ${PROJECT_NAME}::multistartgpu)
endforeach()

#-------------------------------------------------------------------------------
# Build tests
#-------------------------------------------------------------------------------
file(GLOB lgv_package_TESTScpp "app/test/*.cpp")
add_executable(${PROJECT_NAME}_test "${lgv_package_TESTScpp}")
target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME}::common ${PROJECT_NAME}::heuristic ${PROJECT_NAME}::multistartgpu)