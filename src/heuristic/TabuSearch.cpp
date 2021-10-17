#include "lgvRouting/heuristic/TabuSearch.h"

using namespace lgv::heuristic;

TabuSearch::TabuSearch(){
    mNumSwap = 0;
    mIteration = 0.0f;
}

TabuSearch::~TabuSearch(){

}

bool 
TabuSearch::initChild(YAML::Node& aNode){
    mNumSwap        = lgv::common::YAMLgetConf<int>(aNode["TabuSearch"], "swap", 2);
    mIteration      = lgv::common::YAMLgetConf<uint64_t>(aNode["TabuSearch"], "iteration", 1000);
    mDiffCost       = lgv::common::YAMLgetConf<float>(aNode["TabuSearch"], "diffCost", 1);
    mListMaxLenght  = lgv::common::YAMLgetConf<int>(aNode["TabuSearch"], "listLenght", 0);
    mTimeout        = lgv::common::YAMLgetConf<uint64_t>(aNode["TabuSearch"], "timeout", 10) * 1e6;
    return true;
}

void 
TabuSearch::runChild(){
    //Initial solution
    lgv::data::Solution start, found;
    found = start = mFinder.FindInitialSolution(*mProblem);
    mFinder.FillReturnMission(found);
    mSolution = found;
    mTabuList.clear();

    //Setting params
    timeStamp_t time = 0;
    for(int i = 0;i < mIteration; i++){
        mTime.tic();

        //Make swap
        lgv::data::Solution newSol = start;
        newSol.makeSwap(mNumSwap);

        if(!isInTabu(newSol)){
            //Check feasibilty of solution founded
            lgv::data::Solution complete = newSol;
            mFinder.FillReturnMission(complete);
            if(found.mCost > complete.mCost + mDiffCost){
                found = complete;
                start = newSol;
            }
            if(found.mCost < mSolution.mCost)
                mSolution = found;
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
TabuSearch::isInTabu(lgv::data::Solution &aResult){
    bool found = true;
    for_each(mTabuList.begin(), mTabuList.end(), [&](lgv::data::Solution& s){
        for(int i = 0 ; i < s.mSolution.size() && found; i++)
            //if(!(s.mSolution[i].mStart.getId() == aResult.mSolution[i].mStart.getId() && s.mSolution[i].mEnd.getId() == aResult.mSolution[i].mEnd.getId())){
            //    found = false;
            //}
            if(s.mSolution[i].mVeh != aResult.mSolution[i].mVeh)
                found = false;
    });
    if(mTabuList.size() == 0)
        found = false;

    if(!found)
        mTabuList.push_back(aResult);
    
    if(mListMaxLenght && mTabuList.size() == mListMaxLenght)
        mTabuList.pop_front();

    return found;
}

bool 
TabuSearch::closeChild(){
    return true;
}