#pragma once

#include "lgvRouting/heuristic/Generic.h"

namespace lgv { namespace heuristic {


/**
 * @brief   DepthLocalSearch algorithm. 
 *          Starting from an initial solution, the algorithm make mNumSwap for mIteration, 
 *          taking aslo solution with threshold mDiffCost for escaping from local minimun.
 *          The solutions that are already visited are putted in a tabu list.
 * 
 */
class TabuSearch : public Generic{
    private:
        int mNumSwap;
        uint64_t mIteration;
        float mDiffCost;
        std::vector<lgv::data::Solution> mTabuList;
    public:
        TabuSearch();
        ~TabuSearch();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
        bool isInTabu(lgv::data::Solution &aResult);
    private:
};

}}