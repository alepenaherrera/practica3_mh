#include <problem.h>
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

using Constraint = pair<int,int>; 
using ConstrMat = vector<vector<int>>; // Constraints matrix
using Instance = vector<double>;
using Dataset = vector<Instance>;
using ProblemInt = Problem<int>;

/**
 * Loads a dataset from a CSV file. 
 * Each line corresponds to an instance, and each value is an attribute.
 * The values are expected to be separated by commas.
 * @param path The path to the CSV file containing the dataset.
 * @return A vector of vectors of doubles, where each inner vector represents an instance with its attributes.
 * @throws runtime_error if the file cannot be opened.
 * */
inline Dataset loadDataset(const string& path) {
    Dataset data;
    ifstream f(path);
    if (!f) throw runtime_error("No se pudo abrir el archivo de datos: " + path);
    string line;
    while (getline(f,line)){
        if (line.empty()) continue;
        Instance i; 
        stringstream ss(line);
        string value; 
        while (getline(ss, value, ',')) {
            i.push_back(stod(value));
        }
        if (!i.empty()) data.push_back(i);
    }
    return data;
}

/**
 * Loads a constraints matrix from a CSV file.
 * Each line corresponds to a row of the matrix, and each value is an integer representing the constraint between two instances.
 * The values are expected to be separated by commas.
 * @param path The path to the CSV file containing the constraints matrix.
 * @return A vector of vectors of integers representing the constraints matrix.
 * @throws runtime_error if the file cannot be opened.
 */
inline ConstrMat loadConstraints(const string& path) {
    ConstrMat constraints;
    ifstream f(path);
    if (!f) throw runtime_error("No se pudo abrir el archivo de restricciones: " + path);
    string line;
    while (getline(f,line)){
        if (line.empty()) continue;
        vector<int> row;
        stringstream ss(line);
        string value;
        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }
        if (!row.empty()) constraints.push_back(row);
    }
    return constraints;
}

/**
 * Calculates the Euclidean distance between two instances represented as vectors of doubles.
 * @param a The first instance as a vector of doubles.
 * @param b The second instance as a vector of doubles.
 * @return The Euclidean distance between the two instances.
 */
inline double euclidean(const Instance &a, const Instance &b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

/**
 * PARProblem class represents a clustering problem with pairwise constraints (Must-Link and Cannot-Link).
 * 
 * @author Alejandra Peñaherrera
 */
class PARProblem : public ProblemInt {
private:
    int k;                          // number of clusters
    int n;                          // number of instances
    int d;                          // number of attributes per instance

    Dataset X;                      // matrix of instances n x m (m attributes)
    ConstrMat MR;                   // matrix of constraints n x n
    vector<Constraint> ML;          // Must-Link pairs
    vector<Constraint> CL;          // Cannot-Link pairsa

    double lambda;

public:

    /**
     * Constructor for the PARProblem class.
     * Loads the dataset and constraints from the specified file paths, and initializes the problem parameters.
     * @param dataPath The path to the CSV file containing the dataset.
     * @param constPath The path to the CSV file containing the constraints matrix.
     * @param k The number of clusters to form.
     */
    PARProblem(const string& dataPath, const string& constPath, int k) {
        this->k = k;
        X = loadDataset(dataPath);
        MR = loadConstraints(constPath);
        n = X.size();
        d = X[0].size();

        // Build ML and CL lists from MR
        double max_dist = 0.0;
        int nConstr = 0; // number of constraints
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++) {
                if (MR[i][j] == 1) {
                    ML.emplace_back(i, j);
                    nConstr++;
                } else if (MR[i][j] == -1) {
                    CL.emplace_back(i, j);
                    nConstr++;
                }
                // Calculate max distance for lambda scaling
                double dist = euclidean(X[i], X[j]);
                if (dist > max_dist) max_dist = dist;
            }  
        // Compute lambda
        lambda = (nConstr > 0) ? max_dist / nConstr : 0.0;
    }

    /**
     * Return the current size of the solution
     */
    size_t getSolutionSize() override { return (size_t)n; }

    /**
     * Return the range of domain of each element of the solution
     */
    pair<int,int> getSolutionDomainRange() override { return {0, k-1}; }

    /**
     * Check if the solution is a valid solution (every cluster has at least one instance)
     * @param solution to check.
     * @return true if the solution is valid, false otherwise.
     */
    bool isValid(const tSolution<int> &solution) override {
        vector<bool> seen(k, false);
        for (int v : solution) seen[v] = true;
        for (int c = 0; c < k; c++) if (!seen[c]) return false;
        return true;
    }
    
    /**
     * Update the solution to fix it, if isValid returns false
     * @param solution to fix.
     */
    void fix(tSolution<int> &solution) override {
        vector<vector<int>> clusters(k);
        for (int i = 0; i < n; i++) clusters[solution[i]].push_back(i);

        for (int c = 0; c < k; c++) {
            if (clusters[c].empty()) {
                // Find the largest cluster to steal an instance from
                int src = 0; 
                for (int c2 = 1; c2 < k; c2++) {
                    if (clusters[c2].size() > clusters[src].size()) {
                        src = c2;
                    }
                }
                // Pick a random element from the source cluster and move it to the empty cluster
                int idx = Random::get<int>(0, (int)clusters[src].size() - 1);
                int inst = clusters[src][idx];

                clusters[src].erase(clusters[src].begin() + idx); // Remove from source cluster
                clusters[c].push_back(inst); // Add to target cluster
                solution[inst] = c; // Move to empty cluster
            }
        }
    }

    /**
     * Calculate the deviation of a solution, defined as the average intra-cluster distance.
     * @param solution to evaluate.
     * @return The deviation of the solution.
     */
    double deviation(const tSolution<int> &solution) {
        // Calculate centroids of each cluster
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

        // Calculate deviation = average of intra-cluster distances
        double deviation = 0.0;
        for (int c = 0; c < k; c++){
            double intra_dist = 0.0;
            for (int i = 0; i < n; i++) {
                if (solution[i] == c) 
                    intra_dist += euclidean(X[i], centroids[c]); 
            }
            if (sizes[c] > 0)  intra_dist /= sizes[c];
            deviation += intra_dist;
        }
        return deviation/k;
    }

    /**
     * Calculate the infeasibility of a solution, defined as the number of violated constraints.
     * A Must-Link constraint is violated if the two instances are in different clusters.
     * A Cannot-Link constraint is violated if the two instances are in the same cluster.
     * @param solution to evaluate.
     * @return The infeasibility of the solution.
     */
    int infeasibility(const tSolution<int> &solution) {
        int inf = 0;
        for (auto &[i,j] : ML) if (solution[i] != solution[j]) inf++;
        for (auto &[i,j] : CL) if (solution[i] == solution[j]) inf++;
        return inf;
    }

    /***
     * Evaluate the solution from scratch.
     * @param solution to evaluate.
     */
    tFitness fitness(const tSolution<int> &solution) override {
        return (tFitness)(deviation(solution) + lambda * infeasibility(solution));
    }

    /**
     * Evaluate the solution indicating the current position to change and the new
     * value.
     *
     * By default it run the previous one. However, it can be override to
     * factorize the fitness.
     *
     * @param solution to evaluate.
     * @param solution_info information of solution to calculate the factored
     * fitness.
     * @param pos_change position of the solution to change.
     * @param new_value to store in pos_previous.
     */
    tFitness fitness(const tSolution<int> &solution,
                           SolutionFactoringInfo<int> *solution_info,
                           unsigned pos_change, int new_value) override {
        tSolution<int> newsol(solution);
        newsol[pos_change] = new_value;
        return fitness(newsol);
    }

    /**
     * Create a new solution.
     * @return A new solution with random cluster assignments, ensuring that no cluster is empty.
     */
    tSolution<int> createSolution() override {
        tSolution<int> sol(n);
    
        vector<int> idxs(n);
        iota(idxs.begin(), idxs.end(), 0); // [0,1,2,...,n-1]

        // Randomly shuffle the indices to assign clusters in random order
        Random::shuffle(idxs);

        // Assign a fixed cluster to the first k instances to ensure no cluster is empty
        for (int c = 0; c < k; c++) sol[idxs[c]] = c ; // clusters 0..k-1
    
        // Then assign random clusters to the remaining instances
        for (int i = k; i < n; i++) sol[idxs[i]] = Random::get<int>(0, k-1);
    
        return sol;
    }

    /**
     * Getters for the problem data and parameters
     */
    const Dataset& getDataset() const { return X; }
    const vector<Constraint>& getML() const { return ML; }
    const vector<Constraint>& getCL() const { return CL; }
    double getLambda() const { return lambda; }
    int getK() const { return k; }
    int getN() const { return n; }
    int getD() const { return d; }
};