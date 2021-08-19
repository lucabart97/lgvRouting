#pragma once

#include "lgvRouting/common/common.h"
#include "lgvRouting/data/Location.h"

namespace lgv { namespace data {
    class Mission{
        public:
            float mCost;
            Location mEnd;
            Location mStart;

            Mission();
            Mission(const Mission& aMission);
            Mission(Location aStart, Location aEnd, float aCost);
            ~Mission();
            bool operator<(const Mission& aMission) const;
            void fillDistance(const DistanceFunction aFunction = DistanceFunction::EUCLIDEAN);
    };
}}