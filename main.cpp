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
#include "randomsearch.h"
#include "genetic.h"
#include "memetic.h"

using namespace std;
using namespace std::chrono;

// ── Stats ─────────────────────────────────────────────────────────────────────
struct Stats {
    double fitness, distance, incumplimientos, evaluaciones, tiempo;
    tSolution<int> lastSolution;
    vector<double> rawFits; // ← for boxplot CSV (remove after generating plots)
};

// ── CSV output for boxplots (remove after generating plots) ───────────────────
void saveCSV(const string &filename,
             const vector<string> &algNames,
             const vector<vector<double>> &allFits) {
    ofstream f(filename);
    f << "alg,fitness\n";
    for (int a = 0; a < (int)algNames.size(); a++)
        for (double fit : allFits[a])
            f << algNames[a] << "," << fit << "\n";
}
// ── end CSV ───────────────────────────────────────────────────────────────────

// ── Runner helpers ────────────────────────────────────────────────────────────
Stats runGreedy(GreedyCOPKM &alg, PARProblem &par,
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
    Stats st = { mean(fits), mean(dists), mean(infs), mean(evals), mean(times), lastSol };
    st.rawFits = fits;
    return st;
}

Stats runAlgorithm(MH<int> &alg, PARProblem &par,
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
    Stats st = { mean(fits), mean(dists), mean(infs), mean(evals), mean(times), lastSol };
    st.rawFits = fits;
    return st;
}

Stats runBL(LocalSearch &alg, PARProblem &par,
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
    Stats st = { mean(fits), mean(dists), mean(infs), mean(evals), mean(times), lastSol };
    st.rawFits = fits;
    return st;
}

Stats runAGG(AGG &alg, PARProblem &par,
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
    Stats st = { mean(fits), mean(dists), mean(infs), mean(evals), mean(times), lastSol };
    st.rawFits = fits;
    return st;
}

Stats runAGE(AGE &alg, PARProblem &par,
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
    Stats st = { mean(fits), mean(dists), mean(infs), mean(evals), mean(times), lastSol };
    st.rawFits = fits;
    return st;
}

Stats runAM(AM &alg, PARProblem &par,
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
    Stats st = { mean(fits), mean(dists), mean(infs), mean(evals), mean(times), lastSol };
    st.rawFits = fits;
    return st;
}

// ── Print table row ───────────────────────────────────────────────────────────
void printRow(const string &name, const Stats &s) {
    cout << fixed << setprecision(6);
    cout << left  << setw(12) << name
         << right << setw(12) << s.fitness
                  << setw(12) << s.distance
                  << setw(12) << s.incumplimientos
                  << setw(14) << (long long)s.evaluaciones
                  << setw(12) << s.tiempo << "\n";
}

// ── Print solution details ────────────────────────────────────────────────────
void printSolution(const string &algName, const tSolution<int> &sol,
                   PARProblem &par) {
    int k = par.getK(), n = par.getN();
    cout << "\n  [" << algName << " solution — last seed]\n";
    cout << "  Assignment: [";
    for (int i = 0; i < n; i++) cout << sol[i] << (i < n-1 ? "," : "");
    cout << "]\n";
    vector<int> sizes(k, 0);
    for (int v : sol) sizes[v]++;
    cout << "  Cluster sizes: ";
    for (int c = 0; c < k; c++) cout << "c" << c << "=" << sizes[c] << " ";
    cout << "\n";
    int mlViol = 0, clViol = 0;
    for (auto &[i,j] : par.getML()) if (sol[i] != sol[j]) mlViol++;
    for (auto &[i,j] : par.getCL()) if (sol[i] == sol[j]) clViol++;
    cout << "  Violated ML=" << mlViol << "  CL=" << clViol
         << "  Total=" << (mlViol+clViol) << "\n";
    double dev = par.deviation(sol);
    int    inf = par.infeasibility(sol);
    cout << fixed << setprecision(6);
    cout << "  Deviation=" << dev
         << "  Infeasibility=" << inf
         << "  Lambda=" << par.getLambda()
         << "  Fitness=" << (dev + inf * par.getLambda()) << "\n";
}

// ── Dataset config ────────────────────────────────────────────────────────────
struct DatasetConfig { string name, dataFile, constFile; int k; };

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    bool verbose = false;
    vector<long> seeds;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--verbose" || arg == "-v")
            verbose = true;
        else
            seeds.push_back(atol(argv[i]));
    }
    if (seeds.empty()) seeds = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    auto fileExists = [](const string &p) {
        ifstream f(p); return f.good();
    };
    const vector<string> requiredFiles = {
        "zoo_set.dat", "zoo_set_const_15.dat", "zoo_set_const_30.dat",
        "glass_set.dat", "glass_set_const_15.dat", "glass_set_const_30.dat",
        "bupa_set.dat", "bupa_set_const_15.dat", "bupa_set_const_30.dat"
    };
    auto dirHasAllRequiredFiles = [&](const string &dir) {
        for (const auto &f : requiredFiles)
            if (!fileExists(dir + "/" + f)) return false;
        return true;
    };
    string dataDir = "data";
    if (!dirHasAllRequiredFiles(dataDir)) {
        string exePath = (argc > 0 && argv[0]) ? string(argv[0]) : string();
        auto pos = exePath.find_last_of("/\\");
        string exeDir = (pos == string::npos) ? "." : exePath.substr(0, pos);
        string exeDataDir = exeDir + "/../data";
        if (dirHasAllRequiredFiles(exeDataDir))
            dataDir = exeDataDir;
        else
            throw runtime_error("No se encontro un data/ valido.");
    }
    auto dp = [&](const string &f) { return dataDir + "/" + f; };

    const vector<DatasetConfig> DATASETS = {
        { "Zoo_15",   dp("zoo_set.dat"),   dp("zoo_set_const_15.dat"),  7  },
        { "Zoo_30",   dp("zoo_set.dat"),   dp("zoo_set_const_30.dat"),  7  },
        { "Glass_15", dp("glass_set.dat"), dp("glass_set_const_15.dat"),7  },
        { "Glass_30", dp("glass_set.dat"), dp("glass_set_const_30.dat"),7  },
        { "Bupa_15",  dp("bupa_set.dat"),  dp("bupa_set_const_15.dat"), 16 },
        { "Bupa_30",  dp("bupa_set.dat"),  dp("bupa_set_const_30.dat"), 16 },
    };

    const int MAX_EVALS = 100000;
    const CrossoverType BEST_AGG = CrossoverType::UNIFORM;

    for (auto &ds : DATASETS) {
        cout << "\n=== " << ds.name << " ===\n";
        try {
            PARProblem par(ds.dataFile, ds.constFile, ds.k);
            cout << "n=" << par.getN() << "  d=" << par.getD()
                 << "  k=" << par.getK()
                 << "  |ML|=" << par.getML().size()
                 << "  |CL|=" << par.getCL().size()
                 << "  lambda=" << fixed << setprecision(6) << par.getLambda() << "\n";

            // Instantiate algorithms
            GreedyCOPKM greedy;
            RandomSearch rnd;
            LocalSearch  bl;
            AGG aggUN(CrossoverType::UNIFORM);
            AGG aggSF(CrossoverType::FIXED_SEGMENT);
            AGE ageUN(CrossoverType::UNIFORM);
            AGE ageSF(CrossoverType::FIXED_SEGMENT);
            AM  amAll (BEST_AGG, MemeticMode::ALL);
            AM  amRand(BEST_AGG, MemeticMode::RAND);
            AM  amBest(BEST_AGG, MemeticMode::BEST);

            // Print header
            cout << left  << setw(12) << "Algoritmo"
                 << right << setw(12) << "Fitness"
                          << setw(12) << "Distancia"
                          << setw(12) << "Incumpl."
                          << setw(14) << "Evaluaciones"
                          << setw(12) << "Tiempo(s)" << "\n";
            cout << string(74, '-') << "\n";

            // Run all algorithms
            auto sGreedy = runGreedy(greedy, par, seeds, MAX_EVALS);
            auto sRnd    = runAlgorithm(rnd,  par, seeds, MAX_EVALS);
            auto sBL     = runBL(bl,          par, seeds, MAX_EVALS);
            auto sAGGUN  = runAGG(aggUN,       par, seeds, MAX_EVALS);
            auto sAGGSF  = runAGG(aggSF,       par, seeds, MAX_EVALS);
            auto sAGEUN  = runAGE(ageUN,       par, seeds, MAX_EVALS);
            auto sAGESF  = runAGE(ageSF,       par, seeds, MAX_EVALS);
            auto sAMAll  = runAM(amAll,        par, seeds, MAX_EVALS);
            auto sAMRand = runAM(amRand,       par, seeds, MAX_EVALS);
            auto sAMBest = runAM(amBest,       par, seeds, MAX_EVALS);

            // Print table
            printRow("Greedy",  sGreedy);
            printRow("Random",  sRnd);
            printRow("BL",      sBL);
            printRow("AGG-UN",  sAGGUN);
            printRow("AGG-SF",  sAGGSF);
            printRow("AGE-UN",  sAGEUN);
            printRow("AGE-SF",  sAGESF);
            printRow("AM-All",  sAMAll);
            printRow("AM-Rand", sAMRand);
            printRow("AM-Best", sAMBest);

            if (verbose) {
                printSolution("Greedy",  sGreedy.lastSolution,  par);
                printSolution("Random",  sRnd.lastSolution,     par);
                printSolution("BL",      sBL.lastSolution,      par);
                printSolution("AGG-UN",  sAGGUN.lastSolution,   par);
                printSolution("AGG-SF",  sAGGSF.lastSolution,   par);
                printSolution("AGE-UN",  sAGEUN.lastSolution,   par);
                printSolution("AGE-SF",  sAGESF.lastSolution,   par);
                printSolution("AM-All",  sAMAll.lastSolution,   par);
                printSolution("AM-Rand", sAMRand.lastSolution,  par);
                printSolution("AM-Best", sAMBest.lastSolution,  par);
            }

            // ── CSV for boxplots (remove after generating plots) ──────────────
            // vector<string> algNames = {"Greedy","Random","BL","AGG-UN","AGG-SF",
            //                            "AGE-UN","AGE-SF","AM-All","AM-Rand","AM-Best"};
            vector<string> algNames = {"BL","AGG-UN","AGG-SF",
                                       "AGE-UN","AGE-SF","AM-All","AM-Rand","AM-Best"};
            vector<vector<double>> csvFits = {
                //sGreedy.rawFits, sRnd.rawFits,   
                sBL.rawFits,
                sAGGUN.rawFits,  sAGGSF.rawFits,
                sAGEUN.rawFits,  sAGESF.rawFits,
                sAMAll.rawFits,  sAMRand.rawFits, sAMBest.rawFits
            };
            saveCSV(ds.name + ".csv", algNames, csvFits);
            // ── end CSV ───────────────────────────────────────────────────────

        } catch (const exception &e) {
            cout << "[SKIPPED] " << e.what() << "\n";
        }
    }
    return 0;
}