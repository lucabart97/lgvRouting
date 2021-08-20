#pragma once

#include "lgvRouting/common/common.h"
#include "lgvRouting/data/Mission.h"

namespace lgv { namespace data {

    /**
     * @brief   Problem class descriptor
     * 
     */
    class Problem
    {
    public:
        Location                mDepot;             //!< not used
        std::vector<Location>   mPickUp;            //!< not used
        std::vector<Location>   mDelivery;          //!< not used
        std::vector<Mission>    mMissions;          //!< mission to solve
        uint32_t                mNumberOfVeichles;  //!< vehicle numbers
        uint32_t                mCapacity;          //!< not used

        Problem();
        ~Problem();

        void fillCosts(const DistanceFunction aFunction = DistanceFunction::EUCLIDEAN);
        friend std::ostream& operator<<(std::ostream& aOstream, const Problem& aProblem){
            aOstream<<"Problem:"<<std::endl;
            aOstream<<"\tMission: "<<aProblem.mMissions.size()<<std::endl;
            aOstream<<"\tVehicles: "<<aProblem.mNumberOfVeichles<<std::endl;
            return aOstream;
        }
    };
}}