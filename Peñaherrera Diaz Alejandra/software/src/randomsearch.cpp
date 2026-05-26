#include <randomsearch.h>
#include <ppar.h>

using namespace std;

ResultMH<int> RandomSearch::optimize(Problem<int> &problem, int maxevals) {
    assert(maxevals > 0);
    PARProblem &par = dynamic_cast<PARProblem&>(problem);

    tSolution<int> best_solution;
    tFitness best_fitness = numeric_limits<float>::max();

    for (int i = 0; i < maxevals; i++) {
        tSolution<int> solution = par.createSolution();
        tFitness fitness = par.fitness(solution);
        if (fitness < best_fitness) {
            best_fitness = fitness;
            best_solution = solution;
        }
    }
    return ResultMH<int>(best_solution, best_fitness, maxevals);
}