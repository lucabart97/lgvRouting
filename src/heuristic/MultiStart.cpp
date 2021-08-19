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
    mNumSwap    = lgv::common::YAMLgetConf<int>(aNode["MultiStart"], "Or-opt", 2);
    mNumStart   = lgv::common::YAMLgetConf<int>(aNode["MultiStart"], "start", 10);
    mIteration  = lgv::common::YAMLgetConf<uint64_t>(aNode["MultiStart"], "iteration", 1000);
    mTimeout    = lgv::common::YAMLgetConf<uint64_t>(aNode["MultiStart"], "timeout", 10) * 1e6;
    return true;
}


void 
MultiStart::runChild(){
    mSolution.mCost = 99999999999;
    std::srand(std::time(nullptr));
    std::vector<std::pair<int,int>> rnd(mNumSwap);
    timeStamp_t time = 0;
    for(int j = 0; j < mNumStart; j++){
        lgv::data::Solution random = mFinder.FindRandomSolution(*mProblem);
        for(int i = 0; i < mIteration; i++){
            mTime.tic();
            //Make swap
            for_each(rnd.begin(), rnd.end(), [&](std::pair<int,int>& r){
                r.first = std::rand()/((RAND_MAX + 1u)/random.mSolution.size()-1);
                r.second = std::rand()/((RAND_MAX + 1u)/random.mSolution.size()-1);
            });
            for_each(rnd.begin(), rnd.end(), [&](std::pair<int,int>& r){
                std::swap(random.mSolution[r.first],random.mSolution[r.second]);
            });

            //Check feasibilty of solution founded
            lgv::data::Solution complete = random;
            mFinder.FillReturnMission(complete);
            complete.fillCost();
            mSolution = mSolution.mCost > complete.mCost ? complete : mSolution;
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