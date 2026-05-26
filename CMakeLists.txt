cmake_minimum_required(VERSION 3.10)
project(mh_p1 CXX)

# 1. Estándar C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Optimización (Release)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 3. Recoger archivos de la carpeta src
file(GLOB C_SOURCES "src/*.cpp")

# 4. Indicar carpetas de cabeceras (.h)
# Asegúrate de que tus carpetas se llamen exactamente así (o cámbialas aquí)
include_directories("common" "inc" "include")

# 5. Crear el ejecutable 
# Usamos main.cpp (raíz) + todos los de src/
add_executable(main "main.cpp" ${C_SOURCES})

# 6. Compatibilidad para Linux (GitHub Actions)
if(UNIX AND NOT APPLE)
    target_link_libraries(main stdc++fs)
endif()