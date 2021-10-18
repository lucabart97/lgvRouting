#include "lgvRouting/heuristic/LocalSearch.h"

using namespace lgv::heuristic;

LocalSearch::LocalSearch(){
}

LocalSearch::~LocalSearch(){

}

bool 
LocalSearch::initChild(YAML::Node& aNode){
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
        lgv::data::Solution best = start;
        lgv::data::Solution bestResult = mSolution;
        for(int i = 0; i < 1000; i++){
            lgv::data::Solution newSol = best;
            newSol.makeSwap(1);
            lgv::data::Solution comp = newSol;
            mFinder.FillReturnMission(comp);
            if(comp.mCost < bestResult.mCost){
                best = newSol;
                bestResult = comp;
            }
        }

        //Check feasibilty of solution founded
        if(mSolution.mCost > bestResult.mCost){
            mSolution = bestResult;
            start = best;
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