#include "lgvRouting/heuristic/LocalSearch.h"

using namespace lgv::heuristic;

LocalSearch::LocalSearch(){
    mNumSwap = 0;
    mIteration = 0.0f;
}

LocalSearch::~LocalSearch(){

}

bool 
LocalSearch::initChild(YAML::Node& aNode){
    mNumSwap    = lgv::common::YAMLgetConf<int>(aNode["LocalSearch"], "swap", 2);
    mIteration  = lgv::common::YAMLgetConf<uint64_t>(aNode["LocalSearch"], "iteration", 1000);
    mTimeout    = lgv::common::YAMLgetConf<uint64_t>(aNode["LocalSearch"], "timeout", 10) * 1e6;
    return true;
}

void 
LocalSearch::runChild(){
    //Initial solution
    lgv::data::Solution start;
    mSolution = start = mFinder.FindInitialSolution(*mProblem);
    mFinder.FillReturnMission(mSolution);

    //Settijng variables
    timeStamp_t time = 0;

    for(bool found = true;found;){
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
        }else{
            found = false;
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
LocalSearch::closeChild(){
    return true;
}