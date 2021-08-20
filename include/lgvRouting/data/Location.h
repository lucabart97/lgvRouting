#pragma once

#include "lgvRouting/common/common.h"

namespace lgv { namespace data {
    class Location
    {
    public:
        uint32_t    mId;
        float       mX, mY;
        uint32_t    mOpeningTW1, mClosingTW1;
        uint32_t    mOpeningTW2, mClosingTW2;
        uint32_t    mServiceTime;
        uint32_t    mDemand;
    public:
         Location();
         Location(int32_t aId, float aX, float aY, uint32_t aDemand, 
                  uint32_t aOpeningTW1 = 0, uint32_t aClosingTW1 = 9999, 
                  uint32_t aOpeningTW2 = 9999, uint32_t aClosingTW2 = 9999, 
                  uint32_t aServiceTime = 0);
        ~Location();

        void setId(const uint32_t aId);
        uint32_t getId() const;

        void setX(const float aX);
        float getX() const;

        void setY(const float aY);
        float getY() const;

        void setDemand(const uint32_t aDemand);
        uint32_t getDemand() const;
    };  


    enum class DistanceFunction {
        EUCLIDEAN,
        MANHATTAN,
        MAXIMUM
    };
    inline float distance(const Location& aLoc1, const Location& aLoc2, const DistanceFunction aFunction = DistanceFunction::EUCLIDEAN){
        switch (aFunction)
        {
        case DistanceFunction::EUCLIDEAN:
            return std::sqrt(std::pow(aLoc1.getX() - aLoc2.getX(), 2) + std::pow(aLoc1.getY() - aLoc2.getY(), 2));
            break;

        case DistanceFunction::MANHATTAN:
            return std::fabs(aLoc1.getX() - aLoc2.getX()) + std::fabs(aLoc1.getY() - aLoc2.getY());
            break;
        
        case DistanceFunction::MAXIMUM:
            return std::max(std::fabs(aLoc1.getX() - aLoc2.getX()), std::fabs(aLoc1.getY() - aLoc2.getY()));
            break;

        default:
            return 0.0f;
            break;
        }
    }
}}