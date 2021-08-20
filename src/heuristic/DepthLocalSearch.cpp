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
    mNumSwap    = lgv::common::YAMLgetConf<int>(aNode["DepthLocalSearch"], "Or-opt", 2);
    mIteration  = lgv::common::YAMLgetConf<uint64_t>(aNode["DepthLocalSearch"], "iteration", 1000);
    mTimeout    = lgv::common::YAMLgetConf<uint64_t>(aNode["DepthLocalSearch"], "timeout", 10) * 1e6;
    return true;
}

void 
DepthLocalSearch::runChild(){
    //Initial solution
    lgv::data::Solution start;
    mProblem->fillCosts();
    mSolution = start = mFinder.FindInitialSolution(*mProblem);
    mFinder.FillReturnMission(mSolution);
    mSolution.fillCost();

    //Setting values
    timeStamp_t time = 0;
    std::srand(std::time(nullptr));
    std::vector<std::pair<int,int>> rnd(mNumSwap);

    for(int i = 0;i < mIteration; i++){
        mTime.tic();

        //Make swap
        lgv::data::Solution newSol = start;
        for_each(rnd.begin(), rnd.end(), [&](std::pair<int,int>& r){
            r.first = std::rand()/(((float)RAND_MAX + 1u)/newSol.mSolution.size()-1);
            r.second = std::rand()/(((float)RAND_MAX + 1u)/newSol.mSolution.size()-1);
        });
        for_each(rnd.begin(), rnd.end(), [&](std::pair<int,int>& r){
            std::swap(newSol.mSolution[r.first],newSol.mSolution[r.second]);
            std::swap(newSol.mSolution[r.first].mVeh,newSol.mSolution[r.second].mVeh);
        });
        lgv::data::Solution complete = newSol;
        mFinder.FillReturnMission(complete);
        complete.fillCost();

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