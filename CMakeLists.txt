cmake_minimum_required(VERSION 3.10)
project(EdgeViTInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Chemins des headers (fichiers .h)
include_directories(/opt/homebrew/include/onnxruntime)

# 2. L'exécutable
add_executable(edge_vit_inference src/cpp/main.cpp)

# 3. Lien vers la bibliothèque dynamique (.dylib)
target_link_libraries(edge_vit_inference /opt/homebrew/lib/libonnxruntime.dylib)