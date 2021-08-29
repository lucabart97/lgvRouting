#pragma once

#include "lgvRouting/heuristic/Generic.h"

namespace lgv { namespace heuristic {


/**
 * @brief   TabuSearch algorithm. 
 *          Starting from an initial solution, the algorithm make mNumSwap for mIteration, 
 *          taking also solution with threshold mDiffCost for escaping from local minimun.
 *          The solutions that are already visited are putted in a tabu list.
 * 
 */
class TabuSearch : public Generic{
    private:
        int mNumSwap;                               //!< number of node swap
        uint64_t mIteration;                        //!< algorithm iterations
        float mDiffCost;                            //!< cost threshold
        std::list<lgv::data::Solution>   mTabuList; //!< tabu list
        int mListMaxLenght;                         //!< tabu max list lenght requested by user
    public:
        TabuSearch();
        ~TabuSearch();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
        bool isInTabu(lgv::data::Solution &aResult);
};

}}