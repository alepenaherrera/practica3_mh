#include <genetic.h>
#include <ppar.h>

using namespace std;

AGE::AGE(CrossoverType ct) : crossType(ct) {}

/**
 * Perform crossover between two parents to produce two children. 
 * The crossover type can be UNIFORM or FIXED_SEGMENT.
 * @tparam CrossoverType The type of crossover to perform (UNIFORM or FIXED_SEGMENT).
 * @param p1 First parent solution.
 * @param p2 Second parent solution.
 * @return A pair of offspring solutions resulting from the crossover.
 */
pair<tSolution<int>, tSolution<int>>
AGE::doCrossover(const tSolution<int> &p1, const tSolution<int> &p2) {
    if (crossType == CrossoverType::UNIFORM)
        return crossoverUniform(p1, p2);
    else
        return crossoverFixedSegment(p1, p2);
}

/**
 * Optimize the PAR problem using a steady-state genetic algorithm (AGE) with the specified crossover type.
 * The algorithm maintains a population of solutions, applies selection, crossover, and mutation to create offspring, 
 * and uses a replacement strategy where only the 2 worst individuals in the population are replaced by the 2 offspring from each crossover
 * @param par The PAR problem instance to optimize.
 * @param maxevals The maximum number of fitness evaluations allowed.
 * @return A ResultMHInt containing the best solution found, its fitness, and the number
 */
ResultMHInt AGE::optimize(PARProblem &par, int maxevals) {
    int k = par.getK();
    const int M  = 50; // Population size
    const double Pm = 0.1; // Mutation probability
    int evals = 0;

    // Initialize population
    vector<tSolution<int>> pop(M); // Population of solutions
    vector<tFitness> fits(M); // Corresponding fitness values
    for (int i = 0; i < M; i++) {
        pop[i]  = par.createSolution();
        fits[i] = par.fitness(pop[i]);
        evals++;
    }

    // Track global best for final return
    tSolution<int> bestSol = pop[0];
    tFitness bestFit = fits[0];
    for (int i = 1; i < M; i++)
        if (fits[i] < bestFit) { bestFit = fits[i]; bestSol = pop[i]; }

    while (evals < maxevals) {
        // Selection of two parents by tournament
        int p1i = tournamentSelect(pop, fits);
        int p2i = tournamentSelect(pop, fits);

        // Crossover to produce two offspring
        auto [c1, c2] = doCrossover(pop[p1i], pop[p2i]);

        // Mutation of offspring with probability Pm
        if (Random::get<double>(0.0, 1.0) < Pm) mutate(c1, k);
        if (Random::get<double>(0.0, 1.0) < Pm) mutate(c2, k);

        // Repair invalid offspring
        if (!par.isValid(c1)) par.fix(c1);
        if (!par.isValid(c2)) par.fix(c2);

        // Evaluate children 
        if (evals >= maxevals) break;
        tFitness f1 = par.fitness(c1); evals++;
        tFitness f2 = f1; // fallback if budget runs out
        if (evals < maxevals) { f2 = par.fitness(c2); evals++; }

        // Replacement of the 2 worst individuals in the population with the 2 offspring if they are better
        int w1 = 0, w2 = 1;
        if (fits[w1] < fits[w2]) swap(w1, w2);
        for (int i = 2; i < M; i++) {
            if (fits[i] > fits[w1])      { w2 = w1; w1 = i; }
            else if (fits[i] > fits[w2])   { w2 = i; }
        }
        if (f1 < fits[w1]) { pop[w1] = c1; fits[w1] = f1; }
        else if (f1 < fits[w2]) { pop[w2] = c1; fits[w2] = f1; }
        if (f2 < fits[w1]) { pop[w1] = c2; fits[w1] = f2; }
        else if (f2 < fits[w2]) { pop[w2] = c2; fits[w2] = f2; }

        // Update global best
        for (int i = 0; i < M; i++)
            if (fits[i] < bestFit) { bestFit = fits[i]; bestSol = pop[i]; }
    }

    return ResultMHInt(bestSol, bestFit, (unsigned)evals);
}
