#include <ils.h>
#include <ppar.h>
#include <localsearch.h>
#include <mhtrayectory.h>

using namespace std;

ResultMHInt ILS::optimize(Problem<int> &problem, int maxevals) {
    PARProblem &par = dynamic_cast<PARProblem&>(problem);

    const int NUM_ITER   = 5;
    const int BL_MAXEVAL = 20000;
    int k = par.getK();
    int totalEvals = 0;

    // First BL on random solution (MHTrayectory base handles random init)
    LocalSearch bl;
    tSolution<int> initial = problem.createSolution();
    tFitness fitness = problem.fitness(initial);
    totalEvals += 1; // Count the fitness evaluation of the initial solution
    auto res = bl.optimize(problem, initial, fitness, BL_MAXEVAL);
    totalEvals += res.evaluations; // Count evaluations from BL

    tSolution<int> bestSol = res.solution; // Best solution from first BL
    tFitness bestFit = res.fitness; // Best fitness from first BL

    for (int iter = 1; iter < NUM_ITER; iter++) { // Iterations 2 to 5
        // Mutate best solution
        tSolution<int> mutated = mutateILS(bestSol, k);
        tFitness mutFit = par.fitness(mutated);
        totalEvals++; // Count the fitness evaluation of the mutated solution
        // Apply BL starting from the mutated solution
        LocalSearch bl2;
        auto res2 = bl2.optimize(problem, mutated, mutFit, BL_MAXEVAL);
        totalEvals += res2.evaluations;
        if (res2.fitness < bestFit) {
            bestFit = res2.fitness;
            bestSol = res2.solution;
        }
    }
    return ResultMHInt(bestSol, bestFit, (unsigned)totalEvals);
}