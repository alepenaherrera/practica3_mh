#pragma once
#include <ppar.h>
#include <mh.h>

using namespace std;
using MHInt = MH<int>;
using ProblemInt = Problem<int>;
using ResultMHInt = ResultMH<int>;

/**
 * Iterated Local Search with Simulated Annealing (ILS-ES).
 * Same structure as ILS but uses ES instead of BL as inner optimizer.
 * 5 iterations: first ES on random solution, then ES on mutated best.
 * Each ES capped at 20000 evaluations.
 */
class ILS_ES : public MHInt {
public:
    ILS_ES() {}
    virtual ~ILS_ES() {}
    ResultMHInt optimize(ProblemInt &problem, int maxevals);
};