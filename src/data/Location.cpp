#include "lgvRouting/data/Location.h"

using namespace lgv::data;

Location::Location() : Location(0, 0.0f, 0.0f, 0)
{
}

Location::Location(int32_t aId, float aX, float aY, uint32_t aDemand, 
                   uint32_t aOpeningTW1, uint32_t aClosingTW1, uint32_t aOpeningTW2, 
                   uint32_t aClosingTW2, uint32_t aServiceTime)
{
    this->mId           = aId;
    this->mX            = aX;
    this->mY            = aY;
    this->mDemand       = aDemand;
    this->mOpeningTW1   = aOpeningTW1;
    this->mClosingTW1   = aClosingTW1;
    this->mOpeningTW2   = aOpeningTW2;
    this->mClosingTW2   = aClosingTW2;
    this->mServiceTime  = aServiceTime;
}

Location::~Location()
{
}


void 
Location::setId(const uint32_t aId)
{
    this->mId   = aId;
}

uint32_t 
Location::getId() const
{
    return this->mId;
}

void 
Location::setX(const float aX)
{
    this->mX    = aX;
}

float 
Location::getX() const
{
    return this->mX;
}

void 
Location::setY(const float aY)
{
    this->mY    = aY;
}

float 
Location::getY() const
{
    return this->mY;
}

void 
Location::setDemand(const uint32_t aDemand)
{
    this->mDemand   = aDemand;
}

uint32_t 
Location::getDemand() const
{
    return this->mDemand;
}
