#include "lgvRouting/heuristic/Costructive.h"

using namespace lgv::heuristic;

Costructive::Costructive(){

}

Costructive::~Costructive(){

}

bool 
Costructive::initChild(YAML::Node& aNode){
    return true;
}

void 
Costructive::runChild(){
    mProblem->fillCosts();
    mSolution = mFinder.FindInitialSolutionWithReturn(*mProblem);
}

bool 
Costructive::closeChild(){
    return true;
}