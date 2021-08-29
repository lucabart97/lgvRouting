#include "lgvRouting/data/Dataset.h"
#include "lgvRouting/heuristic/Heuristic.h"

#include <lgvRouting/catch2/catch.hpp>

#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while((void)0, 0)

using namespace lgv::data;

void
checkSolution(lgv::data::Solution& s, std::string& message){
    //Create data
    std::vector<std::vector<MissionResult>> res(s.mNumberOfVeichles);
    for_each(s.mSolution.begin(), s.mSolution.end(), [&](const MissionResult& m){res[m.mVeh].push_back(m);});

    //check solution ordering
    for_each(res.begin(), res.end(), [&](const std::vector<MissionResult>& m){
        for_each(m.begin()+1, m.end(), [&](const MissionResult& m){
            REQUIRE_MESSAGE((&m)[-1].mEnd.mId == m.mStart.mId, message);
        });
    });

    //check solution cost
    //calc cost
    float max = -1;
    for_each(res.begin(), res.end(), [&](std::vector<MissionResult> const & vec){
        float cost = 0.0f;
        for_each(vec.begin(),vec.end(), [&](MissionResult const & m){
            cost += m.mCost;
        });
        max = max < cost ? cost : max;
    });
    REQUIRE_MESSAGE(s.mCost == max, message);
}


TEST_CASE("Test all heuristics on dataset P2") {
    lgv::data::Dataset  d;
    d.load(lgv::data::DatasetType::P2);

    lgv::data::Problem problem;
    lgv::heuristic::Constructive c;

    std::vector<std::string> methods = {"constructive", "localsearch", "multistart", "simulatedannealing",
            "multistartgpu", "multistartmultithreads", "depthlocalsearch", "tabusearch"};
    lgv::heuristic::Methods m;
    lgv::heuristic::init_methods(m);
    YAML::Node conf = YAML::LoadFile(std::string(LGV_PATH) + "/data/conf.yaml");
    for_each(methods.begin(), methods.end(), [&](const std::string& s){m[s]->init(conf);});

    for(int i = 0; i < d.mNumOfInstances-1; i++){
        if(d.loadInstance(problem, i)){
            for_each(methods.begin(), methods.end(), [&](const std::string& s){
                lgv::data::Solution solution = m[s]->run(&problem);
                std::string section_name = s + std::string{"_d"}+std::to_string(i);
                checkSolution(solution, section_name);
            });
        }
        std::cout<<"cicle "<<i<<"/"<<d.mNumOfInstances-1<<std::endl;
    }

    lgv::heuristic::dealloc_methods(m);
}