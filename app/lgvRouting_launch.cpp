#include "lgvRouting/common/CmdParser.h"
#include "lgvRouting/data/Dataset.h"
#include "lgvRouting/heuristic/Heuristic.h"

int main(int argc, char* argv[]) {

    lgv::common::CmdParser cmd(argv, "lgvRouting_launch");
    std::string method = cmd.addOpt("-method", "constructive", "launch method");
    int dataset        = cmd.addIntOpt("-dataset", 10, "dataset id");
    cmd.parse();
    
    //Dataset
    lgv::data::Dataset  d;
    d.load(lgv::data::DatasetType::P2);

    //init data
    lgv::data::Problem problem;
    lgv::heuristic::Methods hMethod;
    lgv::heuristic::init_methods(hMethod);
    YAML::Node conf = YAML::LoadFile(std::string(LGV_PATH) + "/data/conf.yaml");

    std::vector<std::string> methods = {"constructive", "localsearch", "multistart", "simulatedannealing",
            "multistartgpu", "multistartmultithreads", "depthlocalsearch", "tabusearch"};

    if(method == "all"){
        std::vector<std::pair<std::string,std::pair<int,float>>> result;
        if (d.loadInstance(problem, dataset)){
            //init data
            
            for_each(methods.begin(), methods.end(), [&](const std::string& s){hMethod[s]->init(conf);});

            //Fill result
            for_each(methods.begin(), methods.end(), [&](const std::string& s){
                lgv::data::Solution sol = hMethod[s]->run(&problem);
                result.push_back(std::make_pair(s,std::make_pair(sol.mTime,sol.mCost)));
            });

            //order and print
            std::sort(result.begin(), result.end(), 
                [](const std::pair<std::string,std::pair<int,float>>& a, const std::pair<std::string,std::pair<int,float>>& b){
                    return a.second.second < b.second.second;
            });
            for_each(result.begin(), result.end(), [&](const std::pair<std::string,std::pair<int,float>>& v){
                std::cout<<v.first<<":"<<std::string(30-v.first.size(),' ' )<<v.second.second<<"\t"<<v.second.first<<"us"<<std::endl;
            });
        }
    }else{
        if (d.loadInstance(problem, dataset)){
            bool found = false;
            for_each(methods.begin(), methods.end(), [&](const std::string& s){if (s == method) found = true;});
            if (found) {
                hMethod[method]->init(conf);
                lgv::data::Solution solution = hMethod[method]->run(&problem);
                std::cout<<problem<<solution<<std::endl;
            } else {
                lgvERR("Method not found.");
                lgvERR("Please use one of the following:");
                for_each(methods.begin(), methods.end(), [&](const std::string& s){lgvERR("  - "<<s);});
            }
        }
    }
    
    lgv::heuristic::dealloc_methods(hMethod);
    return 0;
}