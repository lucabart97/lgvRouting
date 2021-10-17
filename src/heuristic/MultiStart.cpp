#include "lgvRouting/heuristic/MultiStart.h"

using namespace lgv::heuristic;

MultiStart::MultiStart(){
    mNumSwap = 0;
    mIteration = 0.0f;
}

MultiStart::~MultiStart(){

}

bool 
MultiStart::initChild(YAML::Node& aNode){
    mNumSwap    = lgv::common::YAMLgetConf<int>(aNode["MultiStart"], "swap", 2);
    mNumStart   = lgv::common::YAMLgetConf<int>(aNode["MultiStart"], "start", 10);
    mIteration  = lgv::common::YAMLgetConf<uint64_t>(aNode["MultiStart"], "iteration", 1000);
    mTimeout    = lgv::common::YAMLgetConf<uint64_t>(aNode["MultiStart"], "timeout", 10) * 1e6;
    return true;
}


void 
MultiStart::runChild(){
    //Setting values
    mSolution.mCost = 99999999999;
    timeStamp_t time = 0;

    for(int j = 0; j < mNumStart; j++){
        lgv::data::Solution random = mFinder.FindRandomSolution(*mProblem);
        for(int i = 0; i < mIteration; i++){
            mTime.tic();

            //Make swap
            random.makeSwap(mNumSwap);
            lgv::data::Solution complete = random;
            mFinder.FillReturnMission(complete);

            //Check feasibilty of solution founded
            mSolution = mSolution.mCost > complete.mCost ? complete : mSolution;

            //Timeout
            time += mTime.toc();
            if(time > mTimeout){
                lgvWRN("timeout");
                j = mNumStart;
                i = mIteration;
            }
        }
    }
    mSolution.mTime = time;
}

bool 
MultiStart::closeChild(){
    return true;
}