#pragma once

#include "lgvRouting/common/common.h"
#include "lgvRouting/data/Location.h"

namespace lgv { namespace data {

    /**
     * @brief   Mission data
     * 
     */
    class Mission{
        public:
            float mCost;        //!< cost of the mission
            Location mEnd;      //!< pickup location
            Location mStart;    //!< delivery location

            Mission();
            Mission(const Mission& aMission);
            Mission(Location aStart, Location aEnd, float aCost);
            ~Mission();
            bool operator<(const Mission& aMission) const;
            bool operator>(const Mission& aMission) const;
            void fillDistance(const DistanceFunction aFunction = DistanceFunction::EUCLIDEAN);
    };
}}