cmake_minimum_required(VERSION 3.10)
project(EdgeViTInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Recherche OpenCV de manière propre
find_package(OpenCV REQUIRED)

# 2. Inclusion des headers
include_directories(
    /opt/homebrew/include/onnxruntime
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(edge_vit_inference src/cpp/main.cpp)

# 3. Liens vers les binaires
target_link_libraries(edge_vit_inference 
    /opt/homebrew/lib/libonnxruntime.dylib
    ${OpenCV_LIBS}
)

# Petit message de debug pour confirmer la version d'OpenCV trouvée
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")