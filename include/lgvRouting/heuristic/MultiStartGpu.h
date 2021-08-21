#pragma once

#include "lgvRouting/heuristic/Generic.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>

static void inline lgvCudaHandleError( cudaError_t err,
                        const char *file,
                        int line ) {
    if (err != cudaSuccess) {
        lgvERR(cudaGetErrorString( err )<<"\n");
        lgvERR("file: "<<file<<":"<<line<<"\n");
        
        if(err == cudaErrorUnknown){
            lgvERR("Maybe compiled with wrong sm architecture");
        }

        throw std::runtime_error("cudaError");
    }
}
#define lgvCUDA( err ) (lgvCudaHandleError( err, __FILE__, __LINE__ ))

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(-1);}} while(0)

namespace lgv { namespace heuristic {


/**
 * @brief   MultiStartGpu algorithm. 
 *          Startring from an Nst random points, for each the algorithm make Nsw swap 
 *          for Ni iterations.
 * 
 */
class MultiStartGpu : public Generic{
    private:
        int mNumStart;          //!< starting points
        int mNumSwap;           //!< number of node swap
        uint64_t mIteration;    //!< iteration for each starting point
    private:
        int d_start_size;                               //!< device mission size
        lgv::data::MissionResult* d_start = nullptr;    //!< device mission result
        std::vector<lgv::data::MissionResult> h_start;  //!< host mission result

        curandGenerator_t gen;      //!< gpu random generator
        int d_random_size;          //!< size of gpu array with random values
        float* d_random = nullptr;    //!< gpu array with random values

    public:
        MultiStartGpu();
        ~MultiStartGpu();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
    private:
        void fillStartSolutions();
        void fillBestSolution();
};

void launch_kernel(int aStartNum, lgv::data::MissionResult* aStartSol, int aSwap, int aIteration, int aNumMission, float* aRandom, int aNumberOfVeichles);

__device__ void swapLocation(lgv::data::Location* a, lgv::data::Location* b);

template <typename T> __device__ void inline cudaSwap(T& a, T& b){
    T c(a); a=b; b=c;
}

__device__ float makeCost(lgv::data::MissionResult* aMission, int aNumMission, int aNumberOfVeichles);

__device__ float distanceCuda(lgv::data::Location& aLoc1, lgv::data::Location& aLoc2);

__global__ void kernel_multi_start_gpu(lgv::data::MissionResult* aStart, int aStartNum, int aSwap, int aIteration, int aNumMission, float* aRandom, int aNumberOfVeichles); 

}}