#include "lgvRouting/data/Mission.h"

using namespace lgv::data;

Mission::Mission(){
    mCost = 0.0f;
}

Mission::Mission(const Mission& aMission){
    mCost = aMission.mCost;
    mStart = aMission.mStart;
    mEnd = aMission.mEnd;
}

Mission::Mission(Location aStart, Location aEnd, float aCost){
    mCost = aCost;
    mStart = aStart;
    mEnd = aEnd;
}

Mission::~Mission(){

}

bool 
Mission::operator<(const Mission& aMission) const{
        return (mCost < aMission.mCost);
}

bool 
Mission::operator>(const Mission& aMission) const{
        return (mCost > aMission.mCost);
}


void 
Mission::fillDistance(const DistanceFunction aFunction){
    mCost = lgv::data::distance(mStart, mEnd, aFunction);
}