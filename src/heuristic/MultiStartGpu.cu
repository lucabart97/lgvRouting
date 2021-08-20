#include "lgvRouting/heuristic/MultiStartGpu.h"
#include <curand.h>
#include <curand_kernel.h>

void 
lgv::heuristic::launch_kernel(int aStartNum, lgv::data::MissionResult* aStartSol, int aSwap, int aIteration, int aNumMission){
    kernel_multi_start_gpu<<<ceil(aStartNum/1024),1024>>>(time(NULL), aStartSol, aStartNum, aSwap, aIteration, aNumMission);
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
lgv::heuristic::makeCost(lgv::data::MissionResult* aMission, int aNumMission){
    float max = -99999;
    for(int veh = 0; veh < 10; veh++){
        float cost = 0;
        bool first = true;
        lgv::data::Location* prec;
        for(int i = 0; i < aNumMission; i++){
            if(aMission[i].mVeh == veh){
                if(!first){
                    cost += distanceCuda(*prec, aMission[i].mStart) + distanceCuda(aMission[i].mStart, aMission[i].mEnd);
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
lgv::heuristic::kernel_multi_start_gpu(unsigned int seed, lgv::data::MissionResult* aStart, int aStartNum, int aSwap, int aIteration, int aNumMission){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id-1 > aStartNum)
        return;

    auto listMission = &aStart[id*aNumMission];
    float dprec = makeCost(listMission,aNumMission);

    curandState_t state;
    curand_init(seed, 0, 0, &state);
    int id1[10];
    int id2[10];
    for(int i = 0; i < aIteration; i++){

        //make swaps
        for(int j = 0; j < aSwap; j++){
            id1[j] = curand(&state) % aNumMission;
            id2[j] = curand(&state) % aNumMission;
            swapLocation(&listMission[id1[j]].mStart,&listMission[id2[j]].mStart);
            swapLocation(&listMission[id1[j]].mEnd,&listMission[id2[j]].mEnd);
        }

        float dSwap = makeCost(listMission,aNumMission);
        if(dprec < dSwap){
            for(int j = 0; j < aSwap; j++){
                swapLocation(&listMission[id1[j]].mStart,&listMission[id2[j]].mStart);
                swapLocation(&listMission[id1[j]].mEnd,&listMission[id2[j]].mEnd);
            }
        }else{
            dprec = dSwap;
        }
    }
}
