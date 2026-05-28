#include <bmb.h>
#include <ppar.h>
#include <simulated_annealing.h>
#include <localsearch.h>
#include <limits>

using namespace std;

ResultMHInt BMB::optimize(ProblemInt &problem, int maxevals) {
    PARProblem &par = dynamic_cast<PARProblem&>(problem);
    const int NUM_ITER   = 5;
    const int BL_MAXEVAL = 20000;
    tSolution<int> bestSol;
    tFitness bestFit = numeric_limits<tFitness>::max();
    int totalEvals = 0;
    for (int iter = 0; iter < NUM_ITER; iter++) {
        tSolution<int> init = par.createSolution();
        tFitness initFit = par.fitness(init);
        LocalSearch bl;
        auto res = bl.optimize(problem, init, initFit, BL_MAXEVAL);
        totalEvals += res.evaluations;
        if (res.fitness < bestFit) { bestFit = res.fitness; bestSol = res.solution; }
    }
    return ResultMHInt(bestSol, bestFit, (unsigned)totalEvals);
}