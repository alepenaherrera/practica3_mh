#include <simulated_annealing.h>
#include <ppar.h>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;

ResultMHInt SimulatedAnnealing::optimize(Problem<int> &problem,
                                          const tSolution<int> &current,
                                          tFitness initFitness,
                                          int maxevals) {
    PARProblem &par = dynamic_cast<PARProblem&>(problem);
    int n = par.getN();
    int k = par.getK();

    const double phi = 0.3;
    const double mu = 0.2;
    const double Tf = 1e-3;
    const int max_vecinos = 10 * n;
    const int max_exitos = 1 * n;
    const int M = maxevals / max_vecinos;

    tSolution<int> s = current;
    tFitness fs = initFitness;
    int numEval = 0;

    tSolution<int> bestSol = s;
    tFitness bestFit = fs;

    double T0 = (mu * (double)fs) / (-log(phi));
    if (T0 <= Tf) T0 = Tf * 10.0;
    double T = T0;
    double beta = (M > 0) ? (T0 - Tf) / ((double)M * T0 * Tf) : 0.0;

    vector<int> clusterSize(k, 0);
    for (int c : s) clusterSize[c]++;

    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);

    int nExitos = 0;
    do {
        nExitos      = 0;
        int nVecinos = 0;
        int idx      = 0;
        Random::shuffle(indices);

        while (nExitos < max_exitos && nVecinos < max_vecinos && numEval < maxevals) {
            int i   = indices[idx % n];
            idx++;

            int cur = s[i];
            // if (clusterSize[cur] <= 1) continue;
            if (clusterSize[cur] <= 1) {
                nVecinos++;
                continue;   
            }

            int offset = Random::get<int>(1, k - 1);
            int tgt    = (cur + offset) % k;

            tFitness fs2 = par.fitness(s, nullptr, i, tgt);
            nVecinos++;
            numEval++;

            double delta = (double)fs2 - (double)fs;

            if (delta < 0 || Random::get<double>(0.0, 1.0) < exp(-delta / T)) {
                clusterSize[cur]--;
                clusterSize[tgt]++;
                s[i] = tgt;
                fs   = fs2;
                nExitos++;
                if (fs < bestFit) { bestFit = fs; bestSol = s; }
            }
        }

        T = T / (1.0 + beta * T);
        if (T < Tf) T = Tf;
        
    } while (numEval < maxevals && nExitos != 0);

    return ResultMHInt(bestSol, bestFit, (unsigned)numEval);
}