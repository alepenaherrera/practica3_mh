#pragma once
#include <ppar.h>
#include <mh.h>
#include <mhtrayectory.h>

using namespace std;
using ResultMHInt = ResultMH<int>;
using ProblemInt = Problem<int>;
using MHTrayectoryInt = MHTrayectory<int>;

/**
 * Basic Multi-Start Search (Búsqueda Multiarranque Básica).
 * Generates 5 random solutions, applies BL to each (max 20000 evals each),
 * returns the best solution found.
 */
using MHInt = MH<int>;
class BMB : public MHInt {
public:
    BMB() {}
    virtual ~BMB() {}
    ResultMHInt optimize(ProblemInt &problem, int maxevals);
};