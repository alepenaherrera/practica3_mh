#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <limits>
#include <fstream>
#include <algorithm>

#include "random.hpp"
#include "ppar.h"
#include "greedy_copkm.h"
#include "localsearch.h"
#include "simulated_annealing.h"
#include "bmb.h"
#include "ils.h"
#include "ils_es.h"

using namespace std;
using namespace std::chrono;

struct Stats {
    double fitness, distance, incumplimientos, evaluaciones, tiempo;
    tSolution<int> lastSolution;
    vector<double> rawFits;
};

Stats runAlg(MH<int> &alg, PARProblem &par,
             const vector<long> &seeds, int maxevals) {
    int runs = (int)seeds.size();
    int totalConstr = (int)(par.getML().size() + par.getCL().size());
    vector<double> fits(runs), dists(runs), infs(runs), evals(runs), times(runs);
    tSolution<int> lastSol;
    for (int r = 0; r < runs; r++) {
        Random::seed(seeds[r]);
        auto t0  = high_resolution_clock::now();
        auto res = alg.optimize(par, maxevals);
        auto t1  = high_resolution_clock::now();
        fits[r]  = res.fitness;
        dists[r] = par.deviation(res.solution);
        infs[r]  = (totalConstr > 0)
                   ? (double)par.infeasibility(res.solution) / totalConstr : 0.0;
        evals[r] = (double)res.evaluations;
        times[r] = duration<double>(t1 - t0).count();
        lastSol  = res.solution;
    }
    auto mean = [&](const vector<double> &v) {
        double s = 0; for (auto x : v) s += x; return s / runs;
    };
    Stats st = { mean(fits), mean(dists), mean(infs),
                 mean(evals), mean(times), lastSol };
    st.rawFits = fits;
    return st;
}

Stats runGreedy(GreedyCOPKM &alg, PARProblem &par,
                const vector<long> &seeds, int maxevals) {
    return runAlg(alg, par, seeds, maxevals);
}

void saveCSV(const string &filename,
             const vector<string> &algNames,
             const vector<vector<double>> &allFits) {
    ofstream f(filename);
    f << "alg,fitness\n";
    for (int a = 0; a < (int)algNames.size(); a++)
        for (double fit : allFits[a])
            f << algNames[a] << "," << fit << "\n";
}

void printRow(const string &name, const Stats &s) {
    cout << fixed << setprecision(6);
    cout << left  << setw(12) << name
         << right << setw(12) << s.fitness
                  << setw(12) << s.distance
                  << setw(12) << s.incumplimientos
                  << setw(14) << (long long)s.evaluaciones
                  << setw(12) << s.tiempo << "\n";
}

struct DatasetConfig { string name, dataFile, constFile; int k; };

int main(int argc, char *argv[]) {
    vector<long> seeds;
    for (int i = 1; i < argc; i++)
        seeds.push_back(atol(argv[i]));
    if (seeds.empty()) seeds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto fileExists = [](const string &p) { ifstream f(p); return f.good(); };
    const vector<string> requiredFiles = {
        "zoo_set.dat","zoo_set_const_15.dat","zoo_set_const_30.dat",
        "glass_set.dat","glass_set_const_15.dat","glass_set_const_30.dat",
        "bupa_set.dat","bupa_set_const_15.dat","bupa_set_const_30.dat"
    };
    auto dirHasFiles = [&](const string &dir) {
        for (const auto &f : requiredFiles)
            if (!fileExists(dir + "/" + f)) return false;
        return true;
    };
    string dataDir = "data";
    if (!dirHasFiles(dataDir)) {
        string exePath = (argc > 0 && argv[0]) ? string(argv[0]) : string();
        auto pos = exePath.find_last_of("/\\");
        string exeDir = (pos == string::npos) ? "." : exePath.substr(0, pos);
        string alt = exeDir + "/../data";
        if (dirHasFiles(alt)) dataDir = alt;
        else throw runtime_error("No se encontro un data/ valido.");
    }
    auto dp = [&](const string &f) { return dataDir + "/" + f; };

    const vector<DatasetConfig> DATASETS = {
        {"Zoo_15",   dp("zoo_set.dat"),   dp("zoo_set_const_15.dat"),  7 },
        {"Zoo_30",   dp("zoo_set.dat"),   dp("zoo_set_const_30.dat"),  7 },
        {"Glass_15", dp("glass_set.dat"), dp("glass_set_const_15.dat"),7 },
        {"Glass_30", dp("glass_set.dat"), dp("glass_set_const_30.dat"),7 },
        {"Bupa_15",  dp("bupa_set.dat"),  dp("bupa_set_const_15.dat"),16 },
        {"Bupa_30",  dp("bupa_set.dat"),  dp("bupa_set_const_30.dat"),16 },
    };

    const int MAX_EVALS = 100000;

    for (auto &ds : DATASETS) {
        cout << "\n=== " << ds.name << " ===\n";
        try {
            PARProblem par(ds.dataFile, ds.constFile, ds.k);
            cout << "n=" << par.getN() << "  d=" << par.getD()
                 << "  k=" << par.getK()
                 << "  |ML|=" << par.getML().size()
                 << "  |CL|=" << par.getCL().size()
                 << "  lambda=" << fixed << setprecision(6) << par.getLambda() << "\n";

            GreedyCOPKM        greedy;
            LocalSearch        bl;
            SimulatedAnnealing es;
            BMB                bmb;
            ILS                ils;
            ILS_ES             ils_es;

            cout << left  << setw(12) << "Algoritmo"
                 << right << setw(12) << "Fitness"
                          << setw(12) << "Distancia"
                          << setw(12) << "Incumpl."
                          << setw(14) << "Evaluaciones"
                          << setw(12) << "Tiempo(s)" << "\n";
            cout << string(74, '-') << "\n";

            auto sGreedy = runAlg(greedy, par, seeds, MAX_EVALS);
            auto sBL     = runAlg(bl,     par, seeds, MAX_EVALS);
            auto sES     = runAlg(es,     par, seeds, MAX_EVALS);
            auto sBMB    = runAlg(bmb,    par, seeds, MAX_EVALS);
            auto sILS    = runAlg(ils,    par, seeds, MAX_EVALS);
            auto sILS_ES = runAlg(ils_es, par, seeds, MAX_EVALS);

            printRow("Greedy",  sGreedy);
            printRow("BL",      sBL);
            printRow("ES",      sES);
            printRow("BMB",     sBMB);
            printRow("ILS-BL",  sILS);
            printRow("ILS-ES",  sILS_ES);

            // CSV for boxplots
            vector<string> algNames = {"Greedy","BL","ES","BMB","ILS-BL","ILS-ES"};
            vector<vector<double>> csvFits = {
                sGreedy.rawFits, sBL.rawFits, sES.rawFits,
                sBMB.rawFits, sILS.rawFits, sILS_ES.rawFits
            };
            saveCSV(ds.name + ".csv", algNames, csvFits);

        } catch (const exception &e) {
            cout << "[SKIPPED] " << e.what() << "\n";
        }
    }
    return 0;
}