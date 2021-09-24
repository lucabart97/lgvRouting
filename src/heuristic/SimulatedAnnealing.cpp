#include "lgvRouting/heuristic/SimulatedAnnealing.h"

using namespace lgv::heuristic;

SimulatedAnnealing::SimulatedAnnealing(){
    mTemperature = 0.0f;
}

SimulatedAnnealing::~SimulatedAnnealing(){

}

bool 
SimulatedAnnealing::initChild(YAML::Node& aNode){
    mNumSwap            = lgv::common::YAMLgetConf<int>(aNode["SimulatedAnnealing"], "swap", 2);
    mInitialTemperature = lgv::common::YAMLgetConf<double>(aNode["SimulatedAnnealing"], "initalTemperature", 10);
    mCoolingRate        = lgv::common::YAMLgetConf<double>(aNode["SimulatedAnnealing"], "coolingRate", 0.01);
    mMinTemperature     = lgv::common::YAMLgetConf<double>(aNode["SimulatedAnnealing"], "minTemperature", 1);
    mIterTempDec        = lgv::common::YAMLgetConf<uint64_t>(aNode["SimulatedAnnealing"], "iterTempDec", 1000);
    mTimeout            = lgv::common::YAMLgetConf<uint64_t>(aNode["SimulatedAnnealing"], "timeout", 10) * 1e6;
    return true;
}

void 
SimulatedAnnealing::runChild(){
    //Setting values
    lgv::data::Solution start, found; 
    start = mSolution = mFinder.FindRandomSolution(*mProblem);
    mFinder.FillReturnMission(mSolution);
    mTemperature = mInitialTemperature;
    timeStamp_t time = 0;
    std::srand(std::time(nullptr));
    std::vector<std::pair<int,int>> rnd(mNumSwap);
    
    while(mTemperature > mMinTemperature){
        for(int i = 0;i < mIterTempDec; i++){
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

            if(complete.mCost < mSolution.mCost){
                //Accept better solution
                mSolution = complete;
                start = newSol;
            }else{
                //check feasibility
                double probability = pow(M_E, (start.mCost-found.mCost) / mTemperature);
                double randProb = std::rand()/(((float)RAND_MAX + 1u)/1.0);
                if(probability > randProb){
                    start = newSol;
                }
            }

            //Timeout
            time += mTime.toc();
            if(time > mTimeout){
                lgvWRN("timeout");
                i = mIterTempDec;
                mTemperature = -1;
            }
        }
        mTemperature -= mCoolingRate;
    }
    mSolution.mTime = time;
}

bool 
SimulatedAnnealing::closeChild(){
    return true;
}