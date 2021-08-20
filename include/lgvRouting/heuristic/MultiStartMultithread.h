#pragma once

#include "lgvRouting/heuristic/Generic.h"
#include <omp.h>

namespace lgv { namespace heuristic {


/**
 * @brief   MultiStartMultithread algorithm. 
 *          Startring from an Nst random points, for each the algorithm make Nsw swap 
 *          for Ni iterations.
 * 
 */
class MultiStartMultithread : public Generic{
    private:
        int mNumStart;          //!< starting points
        int mNumSwap;           //!< number of node swap
        int mThreads;           //!< threads number
        uint64_t mIteration;    //!< iteration for each starting point
    public:
        MultiStartMultithread();
        ~MultiStartMultithread();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
};

}}