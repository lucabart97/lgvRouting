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
    mNumSwap    = lgv::common::YAMLgetConf<int>(aNode["LocalSearch"], "Or-opt", 2);
    mIteration  = lgv::common::YAMLgetConf<uint64_t>(aNode["LocalSearch"], "iteration", 1000);
    mTimeout    = lgv::common::YAMLgetConf<uint64_t>(aNode["LocalSearch"], "timeout", 10) * 1e6;
    return true;
}

void 
LocalSearch::runChild(){
    //Initial solution
    lgv::data::Solution start;
    mProblem->fillCosts();
    mSolution = start = mFinder.FindInitialSolution(*mProblem);
    mFinder.FillReturnMission(mSolution);
    mSolution.fillCost();

    std::srand(std::time(nullptr));
    std::vector<std::pair<int,int>> rnd(mNumSwap);
    timeStamp_t time = 0;
    for(bool found = true;found;){
        mTime.tic();
        //Make swap
        lgv::data::Solution newSol = start;
        for_each(rnd.begin(), rnd.end(), [&](std::pair<int,int>& r){
            r.first = std::rand()/((RAND_MAX + 1u)/newSol.mSolution.size()-1);
            r.second = std::rand()/((RAND_MAX + 1u)/newSol.mSolution.size()-1);
        });
        for_each(rnd.begin(), rnd.end(), [&](std::pair<int,int>& r){
            std::swap(newSol.mSolution[r.first],newSol.mSolution[r.second]);
        });

        //Check feasibilty of solution founded
        lgv::data::Solution complete = newSol;
        mFinder.FillReturnMission(complete);
        complete.fillCost();
        if(mSolution.mCost > complete.mCost){
            mSolution = complete;
            start = newSol;
        }else{
            found = false;
        }        
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