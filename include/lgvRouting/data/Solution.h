#pragma once
#include <random>
#include "lgvRouting/common/Time.h"
#include "lgvRouting/data/Location.h"

namespace lgv { namespace data {


    /**
     * @brief   Container for mission result in solution
     * 
     */
    class MissionResult{
        public:
            id_lgv   mVeh;      //!< vehicle id that complete this mission
            Location mStart;    //!< pickup location
            Location mEnd;      //!< delivery location
            float mCost;        //!< cost 
            MissionResult();
            MissionResult(const MissionResult& aMissionResult);
            MissionResult(id_lgv aVeh, Location aStart, Location aEnd);
            MissionResult(id_lgv aVeh, Location aStart, Location aEnd, float aCost);
    };


    /**
     * @brief   Solution descriptor
     * 
     */
    class Solution
    {
    public:
        double mCost;                           //!< solution cost
        timeStamp_t mTime;                      //!< computational solution time
        int  mNumberOfVeichles;                 //!< numer of vehicle used for solution
        std::vector<MissionResult> mSolution;   //!< solution

        void fillCost(){
            //Fill lgv by mission id
            std::vector<std::vector<MissionResult>> lgv(mNumberOfVeichles);
            for_each(mSolution.begin(),mSolution.end(), [&](MissionResult const & m){
                lgv[m.mVeh].push_back(m);
            });

            //calc cost
            mCost = -1;
            for_each(lgv.begin(), lgv.end(), [&](std::vector<MissionResult> const & vec){
                float cost = 0.0f;
                for_each(vec.begin(),vec.end(), [&](MissionResult const & m){
                    cost += m.mCost;
                });
                mCost = mCost < cost ? cost : mCost;
            });
        }

        void makeSwap(int n){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> mission(0, mSolution.size()-1);
            std::uniform_int_distribution<> lgv(0, mNumberOfVeichles-1);
            for(int i = 0; i < n; i++){
                int veh = mission(gen);
                int newLgv;
                do{
                    newLgv = lgv(gen);
                }while(newLgv == mSolution[veh].mVeh);
                mSolution[veh].mVeh = newLgv;
            }
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