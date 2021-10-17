#include "lgvRouting/heuristic/SimulatedAnnealing.h"

using namespace lgv::heuristic;

SimulatedAnnealing::SimulatedAnnealing(){
    mTemperature = 0.0f;
}

SimulatedAnnealing::~SimulatedAnnealing(){

}

bool 
SimulatedAnnealing::initChild(YAML::Node& aNode){
    mNumSwap            = 1;
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
    
    while(mTemperature > mMinTemperature){
        for(int i = 0;i < mIterTempDec; i++){
            mTime.tic();

            //Make swap
            lgv::data::Solution newSol = start;
            newSol.makeSwap(mNumSwap);
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