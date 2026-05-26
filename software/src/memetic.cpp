#include <memetic.h>
#include <ppar.h>

using namespace std;

AM::AM(CrossoverType ct, MemeticMode m) : crossType(ct), mode(m) {}

/**
 * Perform crossover between two parents to produce two children. 
 * The crossover type can be UNIFORM or FIXED_SEGMENT.
 * @tparam CrossoverType The type of crossover to perform (UNIFORM or FIXED_SEGMENT).
 * @param p1 First parent solution.
 * @param p2 Second parent solution.
 * @return A pair of offspring solutions resulting from the crossover.
 */
pair<tSolution<int>, tSolution<int>>
AM::doCrossover(const tSolution<int> &p1, const tSolution<int> &p2) {
    if (crossType == CrossoverType::UNIFORM)
        return crossoverUniform(p1, p2);
    else
        return crossoverFixedSegment(p1, p2);
}

/**
 * Select target solutions for applying the local search based on the specified memetic mode.
 * - ALL: Select all solutions in the population.
 * - RAND: Select a random subset of solutions with a 10% chance for each solution.
 * - BEST: Select the top 10% best solutions based on fitness.
 * @param fits The fitness values of the current population.
 * @param M The size of the population.
 * @return A vector of indices corresponding to the selected target solutions.
 */
vector<int> AM::selectTargets(const vector<tFitness> &fits, int M) {
    if (mode == MemeticMode::ALL) {
        vector<int> all(M);
        // Fill with indices 0 to M-1
        iota(all.begin(), all.end(), 0);
        return all;
    } else if (mode == MemeticMode::RAND) {
        vector<int> selected;
        // For each chromosome
        for (int i = 0; i < M; i++)
            // Select with 10% probability
            if (Random::get<double>(0.0, 1.0) < 0.1)
                selected.push_back(i);
        // Guarantee at least one is selected
        if (selected.empty()) selected.push_back(Random::get<int>(0, M - 1));
        return selected;
    } else { // MemeticMode::BEST
        // Get indices of the top 10% best solutions
        int nBest = max(1, (int)(0.1 * M));
        vector<int> indices(M);
        iota(indices.begin(), indices.end(), 0);
        // Sort indices by fitness and keep the top nBest
        sort(indices.begin(), indices.end(),
             [&](int a, int b) { return fits[a] < fits[b]; });
        // Resize to keep only the top nBest indices
        indices.resize(nBest);
        return indices;
    }
}

/**
 * Optimize the PAR problem using a memetic algorithm (AM) with the specified crossover type and memetic mode.
 * The algorithm maintains a population of solutions, applies selection, crossover, and mutation to create offspring,
 * and uses a local search (BLS) to refine selected solutions based on the memetic mode.
 * @param par The PAR problem instance to optimize.
 * @param maxevals The maximum number of fitness evaluations allowed.
 * @return A ResultMHInt containing the best solution found, its fitness, and the number of evaluations used.
 */
ResultMHInt AM::optimize(PARProblem &par, int maxevals) {
    int k = par.getK();
    int n = par.getN();
    const int M = 50; // Population size
    const double Pc = 0.8; // Crossover probability
    const double Pm = 0.1; // Mutation probability
    const int BLS_MAXEVALS = 100;
    const int BLS_PERIOD = 10;
    const int epsilon = max(1, (int)(0.1 * n)); // ξ = 0.1·N
    int evals = 0;
    int generation = 0;

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
        generation++;

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

        // Mutation
        int numMutations = max(1, (int)round(Pm * M));
        for (int m = 0; m < numMutations; m++) {
            int idx = Random::get<int>(0, M - 1);
            mutate(offspring[idx], k);
        }

        // Repair
        for (auto &s : offspring)
            if (!par.isValid(s)) par.fix(s);

        // Evaluate offspring
        vector<tFitness> newFits(M, numeric_limits<tFitness>::max());
        for (int i = 0; i < M && evals < maxevals; i++) {
            newFits[i] = par.fitness(offspring[i]);
            evals++;
        }

        // Elitism
        int newBestIdx = (int)(min_element(newFits.begin(), newFits.end()) - newFits.begin());
        if (bestFit < newFits[newBestIdx]) {
            int worstIdx = (int)(max_element(newFits.begin(), newFits.end()) - newFits.begin());
            offspring[worstIdx] = bestSol;
            newFits[worstIdx]   = bestFit;
        }

        // Replace old population with offspring
        pop  = offspring;
        fits = newFits;

        // Update global best
        for (int i = 0; i < M; i++)
            if (fits[i] < bestFit) { bestFit = fits[i]; bestSol = pop[i]; }

        // Apply BLS to selected targets of the current population every BLS_PERIOD generations
        if (generation % BLS_PERIOD == 0 && evals < maxevals) {
            // Select targets for BLS based on memetic mode
            vector<int> targets = selectTargets(fits, M);
            for (int idx : targets) {
                if (evals >= maxevals) break;
                int blsEvals = min(BLS_MAXEVALS, maxevals - evals);
                // Apply BLS to the selected solution
                auto [newSol, newFit, used] = bls(pop[idx], fits[idx],
                                                   blsEvals, epsilon, par);
                pop[idx] = newSol;
                fits[idx] = newFit;
                evals += used;
                // Update global best if improved
                if (newFit < bestFit) { bestFit = newFit; bestSol = newSol; }
            }
        }
    }

    return ResultMHInt(bestSol, bestFit, (unsigned)evals);
}
