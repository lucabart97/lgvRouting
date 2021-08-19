#pragma once

#include "lgvRouting/common/common.h"
#include "lgvRouting/data/Mission.h"

namespace lgv { namespace data {
    class Problem
    {
    public:
        Location                mDepot;
        std::vector<Location>   mPickUp;
        std::vector<Location>   mDelivery;
        std::vector<Mission>    mMissions;
        uint32_t                mNumberOfVeichles;
        uint32_t                mCapacity;

        Problem();
        ~Problem();

        void fillCosts(const DistanceFunction aFunction = DistanceFunction::EUCLIDEAN);
        friend std::ostream& operator<<(std::ostream& aOstream, const Problem& aProblem){
            aOstream<<"Problem:"<<std::endl;
            aOstream<<"\tMission: "<<aProblem.mMissions.size()<<std::endl;
            aOstream<<"\tVehicles: "<<aProblem.mNumberOfVeichles<<std::endl;
            aOstream<<"\tCapacity: "<<aProblem.mCapacity<<std::endl;
            return aOstream;
        }
    };
}}