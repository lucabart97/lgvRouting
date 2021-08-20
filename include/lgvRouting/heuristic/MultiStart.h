#pragma once

#include "lgvRouting/heuristic/Generic.h"

namespace lgv { namespace heuristic {


/**
 * @brief   MultiStart algorithm. 
 *          Startring from an Nst random points, for each the algorithm make Nsw swap 
 *          for Ni iterations.
 * 
 */
class MultiStart : public Generic{
    private:
        int mNumStart;          //!< starting points
        int mNumSwap;           //!< number of node swap
        uint64_t mIteration;    //!< iteration for each starting point
    public:
        MultiStart();
        ~MultiStart();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
};

}}