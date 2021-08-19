#pragma once

#include "lgvRouting/heuristic/Generic.h"

namespace lgv { namespace heuristic {


/**
 * @brief   DepthLocalSearch algorithm. 
 *          Starting from an initial solution, the algorithm make mNumSwap for mIteration  
 * 
 */
class DepthLocalSearch : public Generic{
    private:
        int mNumSwap;
        uint64_t mIteration;
    public:
        DepthLocalSearch();
        ~DepthLocalSearch();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
    private:
};

}}