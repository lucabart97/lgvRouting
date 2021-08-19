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
        int mNumStart;
        int mNumSwap;
        uint64_t mIteration;
    public:
        MultiStart();
        ~MultiStart();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
};

}}