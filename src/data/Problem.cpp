#include "lgvRouting/data/Problem.h"

using namespace lgv::data;

Problem::Problem()
{
}

Problem::~Problem()
{
}

float 
Problem::distance(const Location& aLoc1, const Location& aLoc2, const DistanceFunction aFunction)
{
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
