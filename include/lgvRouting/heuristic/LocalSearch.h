#pragma once

#include "lgvRouting/heuristic/Generic.h"

namespace lgv { namespace heuristic {


/**
 * @brief   LocalSearch algorithm. 
 *          Starting from an initial solution, the algorithm make mNumSwap until the solution  
 *          not gets better for mIteration
 * 
 */
class LocalSearch : public Generic{
    public:
        LocalSearch();
        ~LocalSearch();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
    private:
};

}}