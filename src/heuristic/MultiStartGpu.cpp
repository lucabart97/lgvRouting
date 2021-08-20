#include "lgvRouting/heuristic/MultiStartGpu.h"

using namespace lgv::heuristic;

MultiStartGpu::MultiStartGpu(){
    d_start_size = 0;
}

MultiStartGpu::~MultiStartGpu(){

}

bool 
MultiStartGpu::initChild(YAML::Node& aNode){
    mNumSwap    = lgv::common::YAMLgetConf<int>(aNode["MultistartGpu"], "Or-opt", 2);
    mNumStart   = lgv::common::YAMLgetConf<int>(aNode["MultistartGpu"], "start", 10);
    mIteration  = lgv::common::YAMLgetConf<uint64_t>(aNode["MultistartGpu"], "iteration", 1000);
    mTimeout    = lgv::common::YAMLgetConf<uint64_t>(aNode["MultistartGpu"], "timeout", 0);
    if(mTimeout != 0)
        lgvWRN("Timeout not possible on gpu");
    return true;
}


void 
MultiStartGpu::runChild(){
    mTime.tic();

    //Check start gpu data
    if(d_start_size == 0 || d_start_size != mNumStart * mProblem->mMissions.size()){
        d_start_size = mNumStart * mProblem->mMissions.size();
        lgvASSERT(d_start_size, "wrong size");
        if(d_start != nullptr){
            lgvCUDA(cudaFree(d_start));
            d_start = nullptr;
        }
        lgvCUDA(cudaMalloc((void**)&d_start, d_start_size*sizeof(lgv::data::MissionResult)));
        h_start.reserve(d_start_size+100);
    }

    //Fill start solution and copy on gpu
    fillStartSolutions();
    lgvCUDA(cudaMemcpy(d_start,h_start.data(),d_start_size*sizeof(lgv::data::MissionResult), cudaMemcpyHostToDevice));

    //Kernel
    launch_kernel(mNumStart, d_start, mNumSwap, mIteration, mProblem->mMissions.size());
    cudaDeviceSynchronize();

    //Copy result on cpu and find the best solution
    lgvCUDA(cudaMemcpy(h_start.data(),d_start,d_start_size*sizeof(lgv::data::MissionResult), cudaMemcpyDeviceToHost));
    fillBestSolution(); 
}

void 
MultiStartGpu::fillStartSolutions(){
    h_start.clear();
    for(int i = 0; i < mNumStart; i++){
        mProblem->fillCosts();
        lgv::data::Solution sol = mFinder.FindRandomSolution(*mProblem);
        for_each(sol.mSolution.begin(), sol.mSolution.end(), [&](const lgv::data::MissionResult& m){
            h_start.push_back(m);
        });
    }
}

void
MultiStartGpu::fillBestSolution(){
    mSolution.mCost = 99999999;
    lgv::data::Solution sol;
    for(int i = 0; i < mNumStart; i++){
        int n = mProblem->mMissions.size();
        sol.mSolution.clear();
        sol.mNumberOfVeichles = mProblem->mNumberOfVeichles;
        for_each(h_start.begin()+i*n, h_start.begin()+(i+1)*n-1,
                [&](const lgv::data::MissionResult& m){
                    sol.mSolution.push_back(m);
                });
        mFinder.FillReturnMission(sol);
        sol.fillCost();
        if(sol.mCost < mSolution.mCost)
            mSolution = sol;
    }
    mSolution.mTime = mTime.toc();
}

bool 
MultiStartGpu::closeChild(){
    if(d_start != nullptr)
        lgvCUDA(cudaFree(d_start));
    return true;
}