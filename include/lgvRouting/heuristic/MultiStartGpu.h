#pragma once

#include "lgvRouting/heuristic/Generic.h"
#include <cuda.h>
#include <cuda_runtime.h>

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

void launch_kernel(int aStartNum, lgv::data::MissionResult* aStartSol, int aSwap, int aIteration, int aNumMission);

__device__ void swapLocation(lgv::data::Location* a, lgv::data::Location* b);

__device__ float makeCost(lgv::data::MissionResult* aMission, int aNumMission);

__device__ float distanceCuda(lgv::data::Location& aLoc1, lgv::data::Location& aLoc2);

__global__ void kernel_multi_start_gpu(unsigned int seed, lgv::data::MissionResult* aStart, int aStartNum, int aSwap, int aIteration, int aNumMission); 

}}