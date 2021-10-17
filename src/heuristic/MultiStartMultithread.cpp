#include "lgvRouting/heuristic/MultiStartMultithread.h"

using namespace lgv::heuristic;

MultiStartMultithread::MultiStartMultithread(){
    mNumSwap = 0;
    mIteration = 0.0f;
}

MultiStartMultithread::~MultiStartMultithread(){

}

bool 
MultiStartMultithread::initChild(YAML::Node& aNode){
    mNumSwap    = lgv::common::YAMLgetConf<int>(aNode["MultiStartMultithread"], "swap", 2);
    mNumStart   = lgv::common::YAMLgetConf<int>(aNode["MultiStartMultithread"], "start", 10);
    mThreads    = lgv::common::YAMLgetConf<int>(aNode["MultiStartMultithread"], "threads", 16);
    mIteration  = lgv::common::YAMLgetConf<uint64_t>(aNode["MultiStartMultithread"], "iteration", 1000);
    mTimeout    = lgv::common::YAMLgetConf<uint64_t>(aNode["MultiStartMultithread"], "timeout", 10) * 1e6;
    return true;
}


void 
MultiStartMultithread::runChild(){
    //Setting values
    mSolution.mCost = 99999999999;
    timeStamp_t time = 0;

    omp_set_num_threads(mThreads);
    #pragma omp parallel for
    for(int j = 0; j < mNumStart; j++){
        lgv::data::Solution random = mFinder.FindRandomSolution(*mProblem);
        for(int i = 0; i < mIteration; i++){
            if(omp_get_thread_num() == 0)
                mTime.tic();

            //Make swap
            random.makeSwap(mNumSwap);
            lgv::data::Solution complete = random;
            mFinder.FillReturnMission(complete);

            //Check feasibilty of solution founded
            mSolution = mSolution.mCost > complete.mCost ? complete : mSolution;

            //Timeout
            if(omp_get_thread_num() == 0){
                time += mTime.toc();
                if(time > mTimeout)
                    lgvWRN("timeout");
            }
            if(time > mTimeout){
                j = mNumStart;
                i = mIteration;
            }
        }
    }
    mSolution.mTime = time;
}

bool 
MultiStartMultithread::closeChild(){
    return true;
}