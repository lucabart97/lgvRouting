#pragma once

#include "lgvRouting/common/common.h"
#include "lgvRouting/data/Location.h"

namespace lgv { namespace data {
    enum class DistanceFunction {
        EUCLIDEAN,
        MANHATTAN,
        MAXIMUM
    };

    class Problem
    {
    public:
        Location                    mDepot;
        std::vector<Location>       mPickUp;
        std::vector<Location>       mDelivery;
        std::vector<std::pair<Location, Location>>  mMissions;
        uint32_t                    mNumberOfVeichles;
        uint32_t                    mCapacity;

        Problem();
        ~Problem();

        float distance(const Location& aLoc1, const Location& aLoc2, const DistanceFunction aFunction = DistanceFunction::EUCLIDEAN);
    };
}}