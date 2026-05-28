#pragma once
#include "mh.h"
#include "mhtrayectory.h"
#include "solution.h"
#include <vector>
#include <numeric>

using namespace std;

using MHInt           = MH<int>;
using ProblemInt      = Problem<int>;
using ResultMHInt     = ResultMH<int>;
using MHTrayectoryInt = MHTrayectory<int>;

/**
 * First-improvement Local Search for PAR.
 * Inherits MHTrayectory so it can be started from a given solution,
 * which is needed by ILS and BMB.
 */
class LocalSearch : public MHTrayectory<int> {
public:
    LocalSearch() {}
    virtual ~LocalSearch() {}

    // Core: BL from a given solution
    ResultMHInt optimize(Problem<int> &problem, const tSolution<int> &current, tFitness fitness, int maxevals);
};