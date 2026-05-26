#pragma once
#include <ppar.h>
#include <genetic.h>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

// @author  Alejandra Peñaherrera

enum class MemeticMode { ALL, RAND, BEST };

/**
 * Soft Local Search (BLS) for PAR. 
 * It explores the same neighborhood as the Local Search, but allows up to ε consecutive non-improving moves before stopping. 
 * @param S Initial solution to improve.
 * @param bestfit Fitness of the initial solution S.
 * @param maxevals Maximum number of evaluations allowed for the local search.
 * @param epsilon Maximum number of consecutive non-improving moves allowed before stopping.
 * @param par The PARProblem instance providing the fitness function and problem details.
 * @return A tuple containing the improved solution, its fitness, and the number of evaluations used
 */
inline tuple<tSolution<int>, tFitness, int>
bls(tSolution<int> S, tFitness bestfit, int maxevals, int epsilon,
    PARProblem &par) {
    int n = par.getN();
    int k = par.getK();
    int evals  = 0;
    int fallos = 0;

    // Build random sequence of indices for exploration
    vector<int> RSI(n);
    iota(RSI.begin(), RSI.end(), 0);
    // Random shuffle of indices for random exploration order
    Random::shuffle(RSI);

    tSolution<int> newsol = S;
    int i = 0;

    // Explore neighborhood until ε consecutive non-improving moves or maxevals reached
    while (fallos < epsilon && i < n && evals < maxevals) {
        
        int p = RSI[i];
        int oldvalue = S[p];

        // Try changing the cluster of instance p to every other cluster
        for (int val = 0; val < k && evals < maxevals; val++) {
            if (val == newsol[p]) continue; // skip if it's alrready in the current cluster

            // Make sure to not leave the current cluster empty
            int cnt = 0;
            for (int j = 0; j < n; j++) if (newsol[j] == newsol[p]) cnt++;
            if (cnt <= 1) continue; // move would leave cluster empty

            // Evaluate the move an its fitness
            newsol[p] = val;
            tFitness fit = par.fitness(newsol);
            evals++;

            // If the move improves fitness, accept it and reset 'fallos'
            if (fit < bestfit) {
                S = newsol;
                bestfit = fit;
            } else {
                newsol[p] = S[p]; // revert
            }
        }

        // If no improving move was found for this instance, count it as a failure
        if (newsol[p] == oldvalue) fallos++;
        i++;
    }

    return {S, bestfit, evals};
}

/**
 * AM: Memetic Algorithm for PAR. It combines a generational AG with a BLS local search applied to selected chromosomes every BLS_PERIOD generations.
 * The selection of chromosomes for local search can be ALL, a random subset (RAND), or the best ones (BEST).
 * @param ct Crossover type (UNIFORM or FIXED_SEGMENT).
 * @param mode Memetic mode for selecting chromosomes for local search (ALL, RAND, BEST).
 * @return A ResultMHInt containing the best solution found, its fitness, and the number of evaluations used.
 */
class AM {
public:
    AM(CrossoverType ct, MemeticMode mode);
    ResultMHInt optimize(PARProblem &par, int maxevals);

private:
    CrossoverType crossType;
    MemeticMode   mode;

    pair<tSolution<int>, tSolution<int>> doCrossover(const tSolution<int> &p1,
                                                      const tSolution<int> &p2);
    vector<int> selectTargets(const vector<tFitness> &fits, int M);
};
