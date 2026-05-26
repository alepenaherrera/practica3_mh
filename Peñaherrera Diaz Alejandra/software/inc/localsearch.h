#pragma once
#include <mh.h>
#include <vector>
#include <numeric>

using namespace std;

// Instantiate the template with the type we are interested in
using MHInt = MH<int>;
using ProblemInt = Problem<int>;
using ResultMHInt = ResultMH<int>;

/**
 * First-improvement Local Search for PAR.
 *
 * Neighbourhood: change cluster of one instance to any other cluster,
 * provided no cluster is left empty.
 * Exploration order: random shuffle of virtual neighbourhood (i, l) pairs.
 * Stopping condition: no improvement found OR maxevals reached.
 * 
 * @author  Alejandra Peñaherrera
 */
class LocalSearch : public MHInt {
public:
    LocalSearch() : MH() {}
    virtual ~LocalSearch() {}

    /**
     * Run the local search algorithm to find a local optimum solution.
     * @param problem The problem to solve.
     * @param maxevals The maximum number of evaluations.
     * @return A pair containing the local optimum solution and its fitness.
     */
    ResultMHInt optimize(ProblemInt &problem, int maxevals);
};