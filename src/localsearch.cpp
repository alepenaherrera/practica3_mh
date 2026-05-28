#include <localsearch.h>
#include <ppar.h>

using namespace std;

ResultMHInt LocalSearch::optimize(ProblemInt &problem,
                                   const tSolution<int> &current,
                                   tFitness fitness,
                                   int maxevals) {
    assert(maxevals > 0);
    PARProblem &par = dynamic_cast<PARProblem&>(problem);
    int n = par.getN();
    int k = par.getK();

    tSolution<int> solution = current;
    tFitness fitness = fitness;
    int evals = 0;

    // Cluster sizes: to make empty-checks O(1)
    vector<int> clusterSize(k, 0);
    for (int c : solution) clusterSize[c]++;

    // Fixed virtual neighborhood: size n * (k - 1)
    // Pair represents: (instance_index, cluster_offset)
    vector<pair<int, int>> neighborhood;
    neighborhood.reserve(n * (k - 1));
    for (int i = 0; i < n; i++)
        for (int offset = 1; offset < k; offset++)
            neighborhood.emplace_back(i, offset);

    bool improvement = true;
    while (improvement && evals < maxevals) {
        improvement = false;
        Random::shuffle(neighborhood);
        for (auto &[i, offset] : neighborhood) {
            if (evals >= maxevals) break;
            int current_cluster = solution[i];
            int target_cluster = (current_cluster + offset) % k;
            if (clusterSize[current_cluster] <= 1) continue;
            tFitness new_fitness = par.fitness(solution, nullptr, i, target_cluster);
            evals++;
            if (new_fitness < fitness) {
                solution[i] = target_cluster;
                clusterSize[current_cluster]--;
                clusterSize[target_cluster]++;
                fitness = new_fitness;
                improvement = true;
                break;
            }
        }
    }
    return ResultMHInt(solution, fitness, (unsigned)evals);
}