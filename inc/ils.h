#pragma once
#include <ppar.h>
#include <mh.h>
#include <vector>
#include <numeric>
#include <mhtrayectory.h>

using namespace std;
using MHInt = MH<int>;
using ProblemInt = Problem<int>;
using ResultMHInt = ResultMH<int>;

// Helper: mutate a solution by changing 20% of genes
inline tSolution<int> mutateILS(const tSolution<int> &sol, int k) {
    int n = (int)sol.size();
    int numChanges = max(1, (int)(0.2 * n));
    tSolution<int> mutated = sol;
    vector<int> clusterSize(k, 0);
    for (int c : mutated) clusterSize[c]++;
    vector<int> positions(n);
    iota(positions.begin(), positions.end(), 0);
    Random::shuffle(positions);
    int changed = 0;
    int idx = 0;
    while (idx < n && changed < numChanges) {
        int i = positions[idx];
        idx++;
        int oldC = mutated[i];
        if (clusterSize[oldC] <= 1) continue;
        int offset = Random::get<int>(1, k - 1);
        int newC = (oldC + offset) % k;
        clusterSize[oldC]--;
        clusterSize[newC]++;
        mutated[i] = newC;
        changed++;
    }
    return mutated;
}

/**
 * Iterated Local Search (Búsqueda Local Reiterada).
 * 5 iterations: first BL on random solution, then BL on mutated best.
 * Mutation changes 20% of genes. Accepts only improvements.
 * Each BL capped at 20000 evaluations.
 */
class ILS : public MHInt {
public:
    ILS() {}
    virtual ~ILS() {}
    ResultMHInt optimize(ProblemInt &problem, int maxevals);
};