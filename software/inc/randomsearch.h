#pragma once
#include <mh.h>
#include <numeric> 

using namespace std;

// Instantiate the template with the type we are interested in
using MHInt = MH<int>;
using ProblemInt = Problem<int>;
using ResultMHInt = ResultMH<int>;

/**
 * Random Search metaheuristic.
 * It generates random solutions until maxevals has been achieved, and returns the best one.
 * 
 * @author  Alejandra Peñaherrera
 */
class RandomSearch : public MHInt {

public:
    RandomSearch() : MH() {}
    virtual ~RandomSearch() {}

    /**
     * Create random solutions until maxevals has been achieved, and returns the
     * best one.
     *
     * @param problem The problem to be optimized
     * @param maxevals Maximum number of evaluations allowed
     * @return A pair containing the best solution found and its fitness
     */
    ResultMHInt optimize(ProblemInt &problem, int maxevals) override;
};
