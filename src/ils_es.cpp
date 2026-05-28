#include <ils_es.h>
#include <ppar.h>
#include <simulated_annealing.h>
#include <mhtrayectory.h>
#include <ils.h>

using namespace std;

ResultMHInt ILS_ES::optimize(Problem<int> &problem, int maxevals) {
    PARProblem &par = dynamic_cast<PARProblem&>(problem);

    const int NUM_ITER   = 5;
    const int ES_MAXEVAL = 20000;
    int k = par.getK();
    int totalEvals = 0;

    // First ES on random solution (MHTrayectory base handles random init)
    SimulatedAnnealing es;
    tSolution<int> initial = problem.createSolution();
    tFitness fitness = problem.fitness(initial);
    totalEvals += 1; // Count the fitness evaluation of the initial solution
    auto res = es.optimize(problem, initial, fitness, ES_MAXEVAL);
    totalEvals += res.evaluations;

    tSolution<int> bestSol = res.solution;
    tFitness bestFit = res.fitness;

    for (int iter = 1; iter < NUM_ITER; iter++) {
        // Mutate best solution
        tSolution<int> mutated = mutateILS(bestSol, k);
        tFitness mutFit = par.fitness(mutated);
        totalEvals++;
        // Apply ES starting from the mutated solution
        SimulatedAnnealing es2;
        auto res2 = es2.optimize(problem, mutated, mutFit, ES_MAXEVAL);
        totalEvals += res2.evaluations;
        if (res2.fitness < bestFit) {
            bestFit = res2.fitness;
            bestSol = res2.solution;
        }
    }
    return ResultMHInt(bestSol, bestFit, (unsigned)totalEvals);
}