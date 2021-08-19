#include "lgvRouting/data/Problem.h"

using namespace lgv::data;

Problem::Problem(){
}

Problem::~Problem(){
}

void 
Problem::fillCosts(const lgv::data::DistanceFunction aFunction){
    for(auto& m : mMissions)
        m.fillDistance(aFunction);
}