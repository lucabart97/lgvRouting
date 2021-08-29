#include "lgvRouting/heuristic/Constructive.h"

using namespace lgv::heuristic;

Constructive::Constructive(){

}

Constructive::~Constructive(){

}

bool 
Constructive::initChild(YAML::Node& aNode){
    mDecreasing = lgv::common::YAMLgetConf<bool>(aNode["Constructive"], "Decreasing", true);
    return true;
}

void 
Constructive::runChild(){
    mProblem->fillCosts();
    mSolution = mFinder.FindInitialSolution(*mProblem,!mDecreasing);
    mFinder.FillReturnMission(mSolution);
}

bool 
Constructive::closeChild(){
    return true;
}