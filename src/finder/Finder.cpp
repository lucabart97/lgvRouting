#include "lgvRouting/finder/Finder.h"

using namespace lgv::data;

Finder::Finder(){

}

Finder::~Finder(){

}

Solution 
Finder::FindInitialSolution(Problem& aProblem){
    lgvASSERT(aProblem.mNumberOfVeichles != 0, "no vehicles");
    lgvASSERT(aProblem.mMissions.size() != 0, "no mission");
    lgvASSERT(aProblem.mMissions.size() > aProblem.mNumberOfVeichles, "no sense");

    std::sort(aProblem.mMissions.begin(), aProblem.mMissions.end(),std::less<Mission>());
    std::vector<float> lgv(aProblem.mNumberOfVeichles,0.0f);

    Solution sol;
    mTime.tic();
    for(auto &p : aProblem.mMissions){
        id_lgv id = std::distance(lgv.begin(), std::min_element(lgv.begin(), lgv.end()));
        lgv[id] += p.mCost;
        sol.mSolution.push_back(MissionResult(id,p.mStart,p.mEnd,p.mCost));
    }
    sol.mTime = mTime.toc();
    sol.mCost = *std::max_element(lgv.begin(), lgv.end());
    sol.mNumberOfVeichles = aProblem.mNumberOfVeichles;
    return sol;
}

Solution 
Finder::FindRandomSolution(Problem& aProblem){
    lgvASSERT(aProblem.mNumberOfVeichles != 0, "no vehicles");
    lgvASSERT(aProblem.mMissions.size() != 0, "no mission");
    lgvASSERT(aProblem.mMissions.size() > aProblem.mNumberOfVeichles, "no sense");

    Solution sol;
    std::srand(std::time(nullptr));
    std::vector<bool> used(aProblem.mMissions.size(), false);
    for(int i = 0; i < aProblem.mMissions.size(); i++){
        int index = std::rand()/(((float)RAND_MAX + 1u))*aProblem.mMissions.size();
        while(used[index])
            index = std::rand()/(((float)RAND_MAX + 1u))*aProblem.mMissions.size();
        used[index] = true;
        auto p = aProblem.mMissions[index];
        sol.mSolution.push_back(MissionResult(i % aProblem.mNumberOfVeichles,p.mStart,p.mEnd,p.mCost));
    }
    sol.mNumberOfVeichles = aProblem.mNumberOfVeichles;
    return sol;
}

void 
Finder::FillReturnMission(Solution& sol){
    std::vector<std::vector<MissionResult>> lgv(sol.mNumberOfVeichles);
    for_each(sol.mSolution.begin(),sol.mSolution.end(), [&](MissionResult const & m){
        lgv[m.mVeh].push_back(m);
    });
    sol.mSolution.clear();
    for_each(lgv.begin(), lgv.end(), [&](std::vector<MissionResult> const & vec){
        float cost = 0.0f;
        for_each(vec.begin()+1,vec.end(), [&](MissionResult const & m){
            sol.mSolution.push_back(MissionResult((&m)[-1].mVeh, 
                (&m)[-1].mStart, (&m)[-1].mEnd, (&m)[-1].mCost));
            lgvASSERT((&m)[-1].mVeh ==  m.mVeh);
            sol.mSolution.push_back(MissionResult(m.mVeh, 
                (&m)[-1].mEnd, m.mStart));
            cost += distance((&m)[-1].mStart, (&m)[-1].mEnd) +
                    distance((&m)[-1].mEnd, m.mStart);
        });
        sol.mCost = sol.mCost < cost ? cost : sol.mCost;
    });
}

Solution 
Finder::FindInitialSolutionWithReturn(Problem& aProblem){
    Solution sol = FindInitialSolution(aProblem);
    FillReturnMission(sol);
    return sol;
}