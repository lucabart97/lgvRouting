#include "lgvRouting/heuristic/MultiStartGpu.h"
#include <curand.h>
#include <curand_kernel.h>

void 
lgv::heuristic::launch_kernel(int aStartNum, lgv::data::MissionResult* aStartSol, int aSwap, int aIteration, int aNumMission, float* aRandom, int aNumberOfVeichles){
    kernel_multi_start_gpu<<<ceil(aStartNum/128+1),128>>>(aStartSol, aStartNum, aSwap, aIteration, aNumMission, aRandom, aNumberOfVeichles);
}

__device__ void 
lgv::heuristic::swapLocation(lgv::data::Location* a, lgv::data::Location* b){
    uint32_t mId = a->mId;
    float    mX  = a->mX;
    float    mY  = a->mY;
    a->mId = b->mId;
    a->mX  = b->mX;
    a->mY  = b->mY;
    b->mId = mId;
    b->mX  = mX;
    b->mY  = mY;
}

__device__ float 
lgv::heuristic::distanceCuda(lgv::data::Location& aLoc1, lgv::data::Location& aLoc2){
    return std::sqrt(std::pow(aLoc1.mX - aLoc2.mX, 2) + std::pow(aLoc1.mY - aLoc2.mY, 2));
}

__device__ float 
lgv::heuristic::makeCost(lgv::data::MissionResult* aMission, int aNumMission, int aNumberOfVeichles){
    float max = -99999;
    for(int veh = 0; veh < aNumberOfVeichles; veh++){
        float cost = 0.0;
        bool first = true;
        lgv::data::Location* prec;
        for(int i = 0; i < aNumMission; i++){
            if(aMission[i].mVeh == veh){
                if(!first){
                    cost += distanceCuda(*prec, aMission[i].mStart) + distanceCuda(aMission[i].mStart, aMission[i].mEnd);
                }else{
                    cost += distanceCuda(aMission[i].mStart, aMission[i].mEnd);
                }
                first = false;
                prec = &aMission[i].mEnd;
            }
        }
        if(cost > max)
            max = cost;
    }
    return max;
}

__global__ void 
lgv::heuristic::kernel_multi_start_gpu(lgv::data::MissionResult* aStart, int aStartNum, int aSwap, int aIteration, int aNumMission, float* aRandom, int aNumberOfVeichles){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= aStartNum)
        return;

    auto random = &aRandom[aSwap*aIteration*id*2];
    auto listMission = &aStart[id*aNumMission];
    float dprec = makeCost(listMission, aNumMission, aNumberOfVeichles);

    int id1[10];
    int id2[10];
    for(int i = 0; i < aIteration; i++){

        //make swaps
        for(int j = 0; j < aSwap; j++){
            id1[j] = random[i*aSwap*2 + j*2 + 0] * (float)(aNumMission-1);
            id2[j] = random[i*aSwap*2 + j*2 + 1] * (float)(aNumMission-1);
            swapLocation(&listMission[id1[j]].mStart,&listMission[id2[j]].mStart);
            swapLocation(&listMission[id1[j]].mEnd,&listMission[id2[j]].mEnd);
            cudaSwap<float>(listMission[id1[j]].mCost,listMission[id2[j]].mCost);
        }

        float dSwap = makeCost(listMission, aNumMission, aNumberOfVeichles);
        if(dprec < dSwap){
            for(int j = aSwap-1; j >= 0; j--){
                swapLocation(&listMission[id1[j]].mStart,&listMission[id2[j]].mStart);
                swapLocation(&listMission[id1[j]].mEnd,&listMission[id2[j]].mEnd);
                cudaSwap<float>(listMission[id1[j]].mCost,listMission[id2[j]].mCost);
            }
        }else{
            dprec = dSwap;
        }
    }
}
