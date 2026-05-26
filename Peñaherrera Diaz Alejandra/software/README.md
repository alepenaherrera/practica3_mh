# Práctica 2 — Metaheurísticas
## Problema de Asignación con Restricciones (PAR)
### Curso 2025-26 — Grado en Ingeniería Informática
## Descripción
Implementación de diez algoritmos para resolver el Problema de Asignación con
Restricciones (PAR): un algoritmo Greedy (COPKM), una Búsqueda Aleatoria (Random),
una Búsqueda Local (BL), cuatro Algoritmos Genéticos (AGG-UN, AGG-SF, AGE-UN, AGE-SF)
y tres Algoritmos Meméticos (AM-All, AM-Rand, AM-Best). El problema consiste en agrupar
un conjunto de instancias en k clusters minimizando la distancia intra-cluster y el
número de restricciones Must-Link y Cannot-Link violadas.

## Estructura del proyecto
- `main.cpp` — Punto de entrada del programa. Carga los seis casos del
  problema, ejecuta los diez algoritmos con las semillas especificadas y
  muestra los resultados en formato tabular.
- `common/` — Framework base independiente del problema.
  - `mh.h` — Interfaz genérica `MH<T>` y estructura `ResultMH<T>`.
  - `problem.h` — Interfaz genérica `Problem<T>`.
  - `solution.h` — Tipos `tSolution<T>` y `tFitness`.
  - `random.hpp` — Generador pseudoaleatorio basado en Mersenne Twister.
  - `util.h` — Utilidades auxiliares.
- `inc/` — Cabeceras específicas del PAR.
  - `ppar.h` — Clase `PARProblem` con toda la lógica del problema.
  - `genetic.h` — Operadores genéticos compartidos (torneo, cruce uniforme,
    cruce por segmento fijo, mutación) y declaraciones de `AGG` y `AGE`.
  - `memetic.h` — Búsqueda Local Suave (`bls`) y declaración de `AM`.
  - `greedy_copkm.h` — Cabecera del algoritmo COPKM.
  - `localsearch.h` — Cabecera de la Búsqueda Local.
  - `randomsearch.h` — Cabecera de la Búsqueda Aleatoria.
- `src/` — Implementaciones de los algoritmos.
  - `genetic_agg.cpp` — Implementación del AGG (generacional con elitismo).
  - `genetic_age.cpp` — Implementación del AGE (estacionario).
  - `memetic.cpp` — Implementación de los Algoritmos Meméticos (AM-All, AM-Rand, AM-Best).
  - `greedy_copkm.cpp` — Implementación del Greedy COPKM.
  - `localsearch.cpp` — Implementación de la Búsqueda Local.
  - `randomsearch.cpp` — Implementación de la Búsqueda Aleatoria.
- `data/` — Ficheros de datos y restricciones de los seis casos del problema.
- `CMakeLists.txt` — Configuración del sistema de construcción CMake.

## Requisitos
- Compilador C++17 o superior
- CMake 3.10 o superior

### macOS (Apple Silicon)
```bash
xcode-select --install
brew install cmake
```
### Linux (Ubuntu/Debian)
```bash
sudo apt install g++ cmake
```

## Compilación
Desde la raíz del proyecto:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
```

## Ejecución
```bash
./build/main [--verbose] [semilla1 semilla2 ... semilla10]
```
El programa ejecuta los diez algoritmos sobre los seis casos del problema
e imprime una tabla de resultados para cada uno. Las semillas se pasan como
argumentos de línea de comandos, lo que permite cambiarlas sin recompilar.

### Ejemplos
```bash
# Ejecución con semillas por defecto (1 al 10)
./build/main
# Ejecución con salida detallada de las soluciones encontradas
./build/main --verbose 1 2 3 4 5 6 7 8 9 10
```

### Opción --verbose
Muestra para cada algoritmo y dataset la solución de la última semilla,
incluyendo el vector de asignación, el tamaño de cada cluster, el número
de restricciones violadas y el desglose del fitness.