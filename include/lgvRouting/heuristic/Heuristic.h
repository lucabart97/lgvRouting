#pragma one
#include <map>

#include "lgvRouting/heuristic/Costructive.h"
#include "lgvRouting/heuristic/LocalSearch.h"
#include "lgvRouting/heuristic/MultiStart.h"
#include "lgvRouting/heuristic/DepthLocalSearch.h"
#include "lgvRouting/heuristic/TabuSearch.h"
#include "lgvRouting/heuristic/SimulatedAnnealing.h"
#include "lgvRouting/heuristic/MultiStartMultithread.h"
#include "lgvRouting/heuristic/MultiStartGpu.h"

namespace lgv { namespace heuristic { 

typedef std::map<std::string,lgv::heuristic::Generic*> Methods;

inline void init_methods(Methods& map){
    map["costructive"]              = new lgv::heuristic::Costructive();
    map["localsearch"]              = new lgv::heuristic::LocalSearch();
    map["multistart"]               = new lgv::heuristic::MultiStart();
    map["multistartgpu"]            = new lgv::heuristic::MultiStartGpu();
    map["multistartmultithreads"]   = new lgv::heuristic::MultiStartMultithread();
    map["depthlocalsearch"]         = new lgv::heuristic::DepthLocalSearch();
    map["tabusearch"]               = new lgv::heuristic::TabuSearch();
    map["simulatedannealing"]       = new lgv::heuristic::SimulatedAnnealing();
}

inline void dealloc_methods(Methods& map){
    for_each(map.begin(),map.end(), [](const std::pair<std::string,lgv::heuristic::Generic*> e){
        delete e.second;
    });
}

}}