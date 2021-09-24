#pragma once

#include "lgvRouting/heuristic/Generic.h"

namespace lgv { namespace heuristic {


/**
 * @brief   SimulatedAnnealing algorithm. 
 *          starting from a random point, perform swap with a probability of P(sol)
 *          where P(sol) = e^(diff(sol, solPrec)/temperature)
 * 
 */
class SimulatedAnnealing : public Generic{
    private:
        int    mNumSwap;            //!< number of node swap
        double mTemperature;        //!< actual temperature
        double mInitialTemperature; //!< initial temperature
        double mCoolingRate;        //!< cooling rate of temperature
        double mMinTemperature;     //!< mininum temperature
        uint64_t mIterTempDec;      //!< iteration to perform before teperature decresing
    public:
        SimulatedAnnealing();
        ~SimulatedAnnealing();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
        bool isInTabu(lgv::data::Solution &aResult);
};

}}