#pragma once

#include "lgvRouting/common/Time.h"
#include "lgvRouting/data/Location.h"

namespace lgv { namespace data {

    class MissionResult{
        public:
            id_lgv   mVeh;
            Location mStart;
            Location mEnd;
            float mCost;
            MissionResult();
            MissionResult(const MissionResult& aMissionResult);
            MissionResult(id_lgv aVeh, Location aStart, Location aEnd);
            MissionResult(id_lgv aVeh, Location aStart, Location aEnd, float aCost);
    };

    class Solution
    {
    public:
        double      mCost;
        timeStamp_t mTime;
        int  mNumberOfVeichles;
        std::vector<MissionResult> mSolution;

        void fillCost(){
            std::vector<float> lgv(mNumberOfVeichles,0.0f);
            for(auto &p : mSolution)
                lgv[std::distance(lgv.begin(), std::min_element(lgv.begin(), lgv.end()))] += p.mCost;
            mCost = *std::max_element(lgv.begin(), lgv.end());
        }

        Solution& operator=(const Solution& s){
            mCost = s.mCost;
            mTime = s.mTime;
            mNumberOfVeichles = s.mNumberOfVeichles;
            mSolution = s.mSolution;
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& aOstream, const Solution& aSolution){
            aOstream<<"Solution:"<<std::endl;
            aOstream<<"\tCost: "<<aSolution.mCost<<std::endl;
            aOstream<<"\tTime: "<<aSolution.mTime<<" ms"<<std::endl;
            for(int i = 0; i < aSolution.mNumberOfVeichles; i++){
                aOstream<<"\tid: "<<i;
                float tot = 0;
                for(auto& s : aSolution.mSolution)
                    if(s.mVeh == i)
                        tot += s.mCost;
                aOstream<<" ["<<tot<<"] ";
                for(auto& s : aSolution.mSolution)
                    if(s.mVeh == i)
                        aOstream<<" ("<<s.mStart.getId()<<","<<s.mEnd.getId()<<")";
                aOstream<<std::endl;
            }
            return aOstream;
        }
    };
}}