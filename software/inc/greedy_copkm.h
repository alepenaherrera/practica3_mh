#pragma once
#include <mh.h>
#include <vector>
#include <numeric>
#include <ppar.h>

using namespace std;

// Instantiate the template with the type we are interested in
using MHInt = MH<int>;
using ProblemInt = Problem<int>;
using ResultMHInt = ResultMH<int>;

/**
 * COPKM Greedy algorithm for PAR.
 *
 * Modified k-means that considers ML/CL restrictions using soft interpretation.
 * For each instance (in random order), assigns it to the cluster that produces
 * the least increase in infeasibility; ties broken by centroid proximity.
 * 
 * @author  Alejandra Peñaherrera
 */
class GreedyCOPKM : public MHInt {

public:
    GreedyCOPKM() : MH() {}
    virtual ~GreedyCOPKM() {}

    /**
     * Builds a solution by making the locally optimal choice at each 
     * step until a complete solution is constructed or maxevals is reached.
     *
     * @param problem The problem to be optimized
     * @param maxevals Maximum number of evaluations allowed
     * @return A pair containing the greedy solution found and its fitness
     */
    ResultMHInt optimize(ProblemInt &problem, int maxevals);

private:
    /**
     * Compute the centroids of each cluster given a solution.
     * @param par The PAR problem instance containing the dataset and number of clusters
     * @param solution The current solution with cluster assignments for each instance
     * @return A vector of centroids, where each centroid is an instance representing the mean of the instances in that cluster
     */
    vector<Instance> computeCentroids(PARProblem &par, const tSolution<int> &solution);
};
