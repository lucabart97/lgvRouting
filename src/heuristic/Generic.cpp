#include "lgvRouting/heuristic/Generic.h"

using namespace lgv::heuristic;

Generic::Generic(){

}

Generic::~Generic(){

}

bool 
Generic::init(YAML::Node& aNode){
    return this->initChild(aNode);
}

lgv::data::Solution 
Generic::run(lgv::data::Problem* aProblem){
    lgvASSERT(aProblem != nullptr)
    mProblem = aProblem;
    mProblem->fillCosts();
    this->runChild();
    return mSolution;
}

bool 
Generic::close(){
    return this->closeChild();
}