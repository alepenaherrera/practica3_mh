#pragma once
#include <ppar.h>
#include <mh.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

using namespace std;

// @author  Alejandro Peñaherrera

using ResultMHInt = ResultMH<int>;

enum class CrossoverType { UNIFORM, FIXED_SEGMENT };

// Shared genetic operators and selection method for AGG and AGE, since they only differ in replacement strategy

/**
 * Tournament selection: randomly pick 3 individuals and return the index of the best one.
 * @param pop The population of solutions.
 * @param fits The fitness values corresponding to the population.
 * @return The index of the selected parent.
 */
inline int tournamentSelect(const vector<tSolution<int>> &pop,
                            const vector<tFitness> &fits) {
    int n = (int)pop.size();
    int a = Random::get<int>(0, n - 1);
    int b = Random::get<int>(0, n - 1);
    int c = Random::get<int>(0, n - 1);
    int best = a;
    if (fits[b] < fits[best]) best = b;
    if (fits[c] < fits[best]) best = c;
    return best;
}

/**
 * Uniform crossover: for each gene, randomly choose with 50% probability whether it comes from parent 1 or parent 2.
 * @param p1 First parent solution.
 * @param p2 Second parent solution.
 * @return A pair of offspring solutions resulting from the crossover.
 */
inline pair<tSolution<int>, tSolution<int>>
crossoverUniform(const tSolution<int> &p1, const tSolution<int> &p2) {
    int n = (int)p1.size();
    tSolution<int> c1(n), c2(n);
    for (int i = 0; i < n; i++) {
        // For each gene, randomly decide whether to swap the genes between the two children
        if (Random::get<double>(0.0, 1.0) < 0.5) {
            c1[i] = p1[i]; c2[i] = p2[i];
        } else {
            c1[i] = p2[i]; c2[i] = p1[i];
        }
    }
    return {c1, c2};
}

/**
 * Fixed-segment crossover: copy a random segment from p1 to c1 and from p2 to c2, then fill the rest of the genes uniformly from both parents.
 * @param p1 First parent solution.
 * @param p2 Second parent solution.
 * @return A pair of offspring solutions resulting from the crossover.
 */
inline pair<tSolution<int>, tSolution<int>>
crossoverFixedSegment(const tSolution<int> &p1, const tSolution<int> &p2) {
    int n = (int)p1.size(); // number of genes (instances)
    int inPos = Random::get<int>(0, n - 1); // random start position for the segment
    int segSize = Random::get<int>(1, max(1, n / 3)); // random segment size 
    int endPos = inPos + segSize - 1;

    // Pre-compute the uniform crossover for non-segment positions
    auto [fill1, fill2] = crossoverUniform(p1, p2);

    tSolution<int> c1(n), c2(n);
    vector<bool> inSeg(n, false);
    for (int s = inPos; s <= endPos; s++) inSeg[s % n] = true;

    for (int i = 0; i < n; i++) {
        // If the position is in the segment, copy from the corresponding parent
        if (inSeg[i]) {
            c1[i] = p1[i];
            c2[i] = p2[i];
        } else {
            // Otherwise, fill from the pre-computed uniform crossover results
            c1[i] = fill1[i];
            c2[i] = fill2[i];
        }
    }
    return {c1, c2};
}

/**
 * Mutate a solution by randomly changing the cluster assignment of one instance to a different cluster, ensuring that no cluster is left empty.
 * @param sol The solution to mutate.
 * @param k The number of clusters (used to determine valid cluster indices).
 */
inline void mutate(tSolution<int> &sol, int k) {
    int n = (int)sol.size();
    int gene = Random::get<int>(0, n - 1); // Randomly select a gene (instance) to mutate
    int oldVal = sol[gene]; // Store the old cluster assignment for the selected gene

    // Don't allow mutation if it would leave the current cluster empty
    int count = 0;
    for (int v : sol) if (v == oldVal) count++;
    if (count <= 1) return;

    // Randomly select a new cluster for the gene, ensuring it's different from the old one
    int newVal = Random::get<int>(0, k - 2);
    if (newVal >= oldVal) newVal++;
    sol[gene] = newVal;
}

// AGG and AGE classes
/**
 * AGG: Generational Genetic Algorithm for PAR. It uses tournament selection, a specified crossover type (UNIFORM or FIXED_SEGMENT), and mutation. The entire population is replaced by the offspring each generation, with elitism to keep the best solution found.
 * @param ct Crossover type (UNIFORM or FIXED_SEGMENT).
 * @return A ResultMHInt containing the best solution found, its fitness, and the number of evaluations used.
 */
class AGG : public MH<int> {
public:
    explicit AGG(CrossoverType ct);
    ResultMHInt optimize(Problem<int> &par, int maxevals);

private:
    CrossoverType crossType;
    pair<tSolution<int>, tSolution<int>> doCrossover(const tSolution<int> &p1,
                                                      const tSolution<int> &p2);
};

/**
 * AGE: Steady-State Genetic Algorithm for PAR. Similar to AGG but with a replacement strategy where only the 2 worst individuals in the population are replaced by the 2 offspring from each crossover, allowing for more gradual evolution.
 * @param ct Crossover type (UNIFORM or FIXED_SEGMENT).
 * @return A ResultMHInt containing the best solution found, its fitness, and the number of evaluations used.
 */
class AGE {
public:
    explicit AGE(CrossoverType ct);
    ResultMHInt optimize(Problem<int> &par, int maxevals);

private:
    CrossoverType crossType;
    pair<tSolution<int>, tSolution<int>> doCrossover(const tSolution<int> &p1,
                                                      const tSolution<int> &p2);
};
