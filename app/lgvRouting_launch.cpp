#include "lgvRouting/common/CmdParser.h"
#include "lgvRouting/data/Dataset.h"
#include "lgvRouting/heuristic/Costructive.h"
#include "lgvRouting/heuristic/LocalSearch.h"
#include "lgvRouting/heuristic/MultiStart.h"
#include "lgvRouting/heuristic/DepthLocalSearch.h"
#include "lgvRouting/heuristic/TabuSearch.h"

int main(int argc, char* argv[]) {

    lgv::common::CmdParser cmd(argv, "lgvRouting_launch");
    std::string method = cmd.addOpt("-method", "costructive", "launch method");
    cmd.parse();
    
    //Dataset
    lgv::data::Dataset  d;
    d.load(lgv::data::DatasetType::P2);

    //init data
    lgv::data::Problem problem;
    lgv::heuristic::Generic* heuristic;
    if(method == "costructive")
        heuristic = new lgv::heuristic::Costructive();
    else if(method == "localsearch")
        heuristic = new lgv::heuristic::LocalSearch();
    else if(method == "multistart")
        heuristic = new lgv::heuristic::MultiStart();
    else if(method == "depthlocalsearch")
        heuristic = new lgv::heuristic::DepthLocalSearch();
    else if(method == "tabusearch")
        heuristic = new lgv::heuristic::TabuSearch();
    else
        lgvFATAL("method not recognized");

    if (d.loadInstance(problem, 10))
    {
        YAML::Node conf = YAML::LoadFile(std::string(LGV_PATH) + "/data/conf.yaml");
        heuristic->init(conf);

        lgv::data::Solution solution = heuristic->run(&problem);
        std::cout<<problem<<solution<<std::endl;
    }
    
    delete heuristic;
    return 0;
}