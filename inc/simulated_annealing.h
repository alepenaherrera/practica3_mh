#pragma once
#include <ppar.h>
#include <mhtrayectory.h>
#include <mh.h>

using namespace std;
using ResultMHInt     = ResultMH<int>;
using ProblemInt      = Problem<int>;
using MHTrayectoryInt = MHTrayectory<int>;

/**
 * Simulated Annealing (Enfriamiento Simulado) for PAR.
 * Modified Cauchy cooling scheme.
 * Parameters: phi=0.3, mu=0.2, Tf=1e-3,
 *             max_vecinos=10*n, max_exitos=1*n
 */
class SimulatedAnnealing : public MHTrayectoryInt {
public:
    SimulatedAnnealing() {}
    virtual ~SimulatedAnnealing() {}

    // Core: ES from a given solution
    ResultMHInt optimize(ProblemInt &problem,
                         const tSolution<int> &current,
                         tFitness fitness,
                         int maxevals) override;
};