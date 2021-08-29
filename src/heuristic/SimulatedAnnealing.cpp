#include "lgvRouting/heuristic/SimulatedAnnealing.h"

using namespace lgv::heuristic;

SimulatedAnnealing::SimulatedAnnealing(){
    mTemperature = 0.0f;
}

SimulatedAnnealing::~SimulatedAnnealing(){

}

bool 
SimulatedAnnealing::initChild(YAML::Node& aNode){
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
    mSolution.mCost = 999999;
    mTemperature = mInitialTemperature;
    timeStamp_t time = 0;
    std::srand(std::time(nullptr));
    
    while(mTemperature > mMinTemperature){
        for(int i = 0;i < mIterTempDec; i++){
            mTime.tic();

            //Make swap
            lgv::data::Solution newSol = start;
            found = newSol = mFinder.FindRandomSolution(*mProblem);
            mFinder.FillReturnMission(newSol);

            if(newSol.mCost < mSolution.mCost){
                //Accept better solution
                mSolution = newSol;
                start = found;
            }else{
                //check feasibility
                double probability = pow(M_E, (-found.mCost) / mTemperature);
                double randProb = std::rand()/(((float)RAND_MAX + 1u)/1.0);
                if(probability > randProb){
                    start = found;
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