#include "lgvRouting/data/Solution.h"

using namespace lgv::data;

MissionResult::MissionResult(){

}

MissionResult::MissionResult(const MissionResult& aMissionResult){
    mVeh = aMissionResult.mVeh;
    mStart = aMissionResult.mStart;
    mEnd = aMissionResult.mEnd;
    mCost = aMissionResult.mCost;
}

MissionResult::MissionResult(id_lgv aVeh, Location aStart, Location aEnd){
    mVeh = aVeh;
    mStart = aStart;
    mEnd = aEnd;
    mCost = distance(aStart,aEnd);
}

MissionResult::MissionResult(id_lgv aVeh, Location aStart, Location aEnd, float aCost){
    mVeh = aVeh;
    mStart = aStart;
    mEnd = aEnd;
    mCost = aCost;
}