#include <genetic.h>
#include <ppar.h>

using namespace std;


AGG::AGG(CrossoverType ct) : crossType(ct) {}

/**
 * Perform crossover between two parents to produce two children. 
 * The crossover type can be UNIFORM or FIXED_SEGMENT.
 * @tparam CrossoverType The type of crossover to perform (UNIFORM or FIXED_SEGMENT).
 * @param p1 First parent solution.
 * @param p2 Second parent solution.
 * @return A pair of offspring solutions resulting from the crossover.
 */
pair<tSolution<int>, tSolution<int>>
AGG::doCrossover(const tSolution<int> &p1, const tSolution<int> &p2) {
    if (crossType == CrossoverType::UNIFORM)
        return crossoverUniform(p1, p2);
    else
        return crossoverFixedSegment(p1, p2);
}

/**
 * Optimize the PAR problem using a generational genetic algorithm (AGG) with the specified crossover type.
 * The algorithm maintains a population of solutions, applies selection, crossover, and mutation to create offspring,
 * and uses elitism to ensure the best solution is preserved across generations.
 * @param par The PAR problem instance to optimize.
 * @param maxevals The maximum number of fitness evaluations allowed.
 * @return A ResultMHInt containing the best solution found, its fitness, and the number of evaluations used.
 */
ResultMHInt AGG::optimize(PARProblem &par, int maxevals) {
    int k = par.getK();
    const int M  = 50; // Population size
    const double Pc = 0.8; // Crossover probability
    const double Pm = 0.1; // Mutation probability
    int evals = 0;

    // Initialize population
    vector<tSolution<int>> pop(M);
    vector<tFitness> fits(M);
    for (int i = 0; i < M; i++) {
        pop[i]  = par.createSolution();
        fits[i] = par.fitness(pop[i]);
        evals++;
    }

    // Track global best for elitism
    int bestIdx = (int)(min_element(fits.begin(), fits.end()) - fits.begin());
    tSolution<int> bestSol = pop[bestIdx];
    tFitness        bestFit = fits[bestIdx];

    while (evals < maxevals) {
        // Selection of parents by tournament
        vector<tSolution<int>> parents(M);
        for (int i = 0; i < M; i++)
            parents[i] = pop[tournamentSelect(pop, fits)];

        // Coss consecutive pairs up to Pc*M
        vector<tSolution<int>> offspring(M);
        int totalCross = (int)round(Pc * M);
        int i = 0;
        if (totalCross % 2 != 0) totalCross--;
        while (i < totalCross) {
            auto [c1, c2] = doCrossover(parents[i], parents[i+1]);
            offspring[i]   = c1;
            offspring[i+1] = c2;
            i += 2;
        }
        // Remaining individuals are copied without crossover
        for (; i < M; i++) offspring[i] = parents[i];

        // Mutation: randomly mutate Pm*M individuals in the offspring
        int numMutations = max(1, (int)round(Pm * M));
        for (int m = 0; m < numMutations; m++) {
            int idx = Random::get<int>(0, M - 1);
            mutate(offspring[idx], k);
        }

        // Repair invalid offspring
        for (auto &s : offspring)
            if (!par.isValid(s)) par.fix(s);

        // Evaluate offspring
        vector<tFitness> newFits(M, numeric_limits<tFitness>::max());
        for (int i = 0; i < M && evals < maxevals; i++) {
            newFits[i] = par.fitness(offspring[i]);
            evals++;
        }
        
        // Elitism: keep best of previous generation if better than best of new generation
        int newBestIdx = (int)(min_element(newFits.begin(), newFits.end()) - newFits.begin());
        if (bestFit < newFits[newBestIdx]) {
            int worstIdx = (int)(max_element(newFits.begin(), newFits.end()) - newFits.begin());
            offspring[worstIdx] = bestSol;
            newFits[worstIdx]   = bestFit;
        }
        // Replace old population with new offspring
        pop  = offspring;
        fits = newFits;

        // Update global best
        int bi = (int)(min_element(fits.begin(), fits.end()) - fits.begin());
        if (fits[bi] < bestFit) {
            bestFit = fits[bi];
            bestSol = pop[bi];
        }
    }

    return ResultMHInt(bestSol, bestFit, (unsigned)evals);
}
