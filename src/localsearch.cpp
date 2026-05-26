#include <localsearch.h>
#include <ppar.h>

using namespace std;

ResultMHInt LocalSearch::optimize(ProblemInt &problem, int maxevals){

    assert(maxevals > 0);

    PARProblem &par = dynamic_cast<PARProblem&>(problem);

    int n = par.getN();
    int k = par.getK();

    // Initial random solution
    tSolution<int> solution = par.createSolution();
    tFitness fitness = par.fitness(solution);
    int evals = 1;

    // Cluster sizes: to make empty-checks O(1)
    vector<int> clusterSize(k, 0);
    for (int c : solution) {
        clusterSize[c]++;
    }

    // Fixed virtual neighborhood: size n * (k - 1)
    // Pair represents: (instance_index, cluster_offset)
    vector<pair<int, int>> neighborhood;
    neighborhood.reserve(n * (k - 1));

    for (int i = 0; i < n; i++) {
        for (int offset = 1; offset < k; offset++) {
            neighborhood.emplace_back(i, offset);
        }
    }

    bool improvement = true;

    while (improvement && evals < maxevals) {
        improvement = false;

        // Shuffle virtual neighborhood to avoid repetitions (Slide 50)
        Random::shuffle(neighborhood);

        for (auto &[i, offset] : neighborhood) {

            if (evals >= maxevals)
                break;

            int current_cluster = solution[i];
            
            // Calculate target cluster using a modular offset
            int target_cluster = (current_cluster + offset) % k;

            // Check if the move would leave the current cluster empty
            if (clusterSize[current_cluster] <= 1) {
                continue; 
            }

            // Evaluate neighbor incrementally
            tFitness new_fitness = par.fitness(solution, nullptr, i, target_cluster);
            evals++;

            // First improvement strategy
            if (new_fitness < fitness) {
                // Update solution
                solution[i] = target_cluster;

                // Update cluster tracking
                clusterSize[current_cluster]--;
                clusterSize[target_cluster]++;

                fitness = new_fitness;
                improvement = true;
                break; // Break the neighbor loop to restart exploration
            }
        }
    }

    return ResultMHInt(solution, fitness, evals);
}