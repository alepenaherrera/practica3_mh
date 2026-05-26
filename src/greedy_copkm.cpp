#include <greedy_copkm.h>
#include <ppar.h>

using namespace std;

vector<Instance> GreedyCOPKM::computeCentroids(PARProblem &par, const tSolution<int> &solution) {
    int k = par.getK();
    int d = par.getD();
    int n = par.getN();
    const Dataset &X = par.getDataset();

    vector<Instance> centroids(k, Instance(d, 0.0));
    vector<int> sizes(k, 0);
    for (int i = 0; i < n; i++) {
        int c = solution[i];
        sizes[c]++;
        // sum of attributes of instances in cluster c
        for (int j = 0; j < d; j++) centroids[c][j] += X[i][j];
    }
    for (int c = 0; c < k; c++) {
        if (sizes[c] > 0) {
            // average of attributes of instances in cluster c
            for (int j = 0; j < d; j++) centroids[c][j] /= sizes[c];
        }
    }
    return centroids;
}

ResultMHInt GreedyCOPKM::optimize(ProblemInt &problem, int maxevals) {
    assert(maxevals > 0);
    PARProblem &par = dynamic_cast<PARProblem&>(problem);
    int n = par.getSolutionSize();
    int k = par.getSolutionDomainRange().second;
    int d = par.getD();
    const Dataset &X = par.getDataset();

    //Generate k random centroids within the range of the dataset attributes
    vector<double> dmin(d,  numeric_limits<double>::max());
    vector<double> dmax(d, -numeric_limits<double>::max());
    for (auto &inst : X) {
        for (int j = 0; j < d; j++) {
            dmin[j] = min(dmin[j], inst[j]);
            dmax[j] = max(dmax[j], inst[j]);
        }
    }
    vector<Instance> centroids(k, Instance(d, 0.0));
    for (int c = 0; c < k; c++) {
        for (int j = 0; j < d; j++) {
            centroids[c][j] = Random::get<double>(dmin[j], dmax[j]);
        }
    } 

    tSolution<int> solution(n,-1);
    int max_iters = 100;
    bool changed = true; 
    int iter = 0;

    vector<int> idxs(n);
    iota(idxs.begin(), idxs.end(), 0); // [0,1,2,...,n-1]


    while (changed && iter < max_iters) {
        changed = false;
        iter++;
        Random::shuffle(idxs);      // Shuffle instance indices
        tSolution<int> new_solution(n);

        // Assign each instance to the closest cluster that violates the least constraints
        for (int idx : idxs) {
            int best_cluster = 0;
            int best_inf = numeric_limits<int>::max();
            double best_dist = numeric_limits<double>::max();

            for (int c = 0; c < k; c++) {
                // Calculate distance to centroid
                double dist = euclidean(X[idx], centroids[c]);

                // Calculate infeasibility if assigned to cluster c
                int inf = 0;
                for (auto &[i,j] : par.getML()) {
                    if (i == idx && solution[j] != c) inf++;
                    else if (j == idx && solution[i] != c) inf++; 
                }
                for (auto &[i,j] : par.getCL()) {
                    if (i == idx && solution[j] == c) inf++;
                    else if (j == idx && solution[i] == c) inf++; 
                }

                if (inf < best_inf) {
                    best_inf = inf;
                    best_dist = dist;
                    best_cluster = c;
                } else if (inf == best_inf) {
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
            }
            new_solution[idx] = best_cluster;
        }
        
        if (new_solution != solution) {
            changed = true;
            solution = new_solution;
            // Update centroids based on the assigned clusters
            centroids = computeCentroids(par, solution);
        }
    }
    // Check if the new solution is valid, if not fix it
    if (!par.isValid(solution)) {
        par.fix(solution);
    }
    tFitness fitness = par.fitness(solution);
    return ResultMHInt(solution, fitness, 1);
}