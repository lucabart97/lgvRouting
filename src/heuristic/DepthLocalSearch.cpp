#include "lgvRouting/heuristic/DepthLocalSearch.h"

using namespace lgv::heuristic;

DepthLocalSearch::DepthLocalSearch(){
    mNumSwap = 0;
    mIteration = 0.0f;
}

DepthLocalSearch::~DepthLocalSearch(){

}

bool 
DepthLocalSearch::initChild(YAML::Node& aNode){
    mNumSwap    = lgv::common::YAMLgetConf<int>(aNode["DepthLocalSearch"], "swap", 2);
    mIteration  = lgv::common::YAMLgetConf<uint64_t>(aNode["DepthLocalSearch"], "iteration", 1000);
    mTimeout    = lgv::common::YAMLgetConf<uint64_t>(aNode["DepthLocalSearch"], "timeout", 10) * 1e6;
    return true;
}

void 
DepthLocalSearch::runChild(){
    //Initial solution
    lgv::data::Solution start;
    mSolution = start = mFinder.FindInitialSolution(*mProblem);
    mFinder.FillReturnMission(mSolution);

    //Setting values
    timeStamp_t time = 0;

    for(int i = 0;i < mIteration; i++){
        mTime.tic();

        //Make swap
        lgv::data::Solution newSol = start;
        newSol.makeSwap(mNumSwap);
        lgv::data::Solution complete = newSol;
        mFinder.FillReturnMission(complete);

        //Check feasibilty of solution founded
        if(mSolution.mCost > complete.mCost){
            mSolution = complete;
            start = newSol;
        }

        //Timeout
        time += mTime.toc();
        if(time > mTimeout){
            lgvWRN("timeout");
            break;
        }
    }
    mSolution.mTime = time;
}

bool 
DepthLocalSearch::closeChild(){
    return true;
}