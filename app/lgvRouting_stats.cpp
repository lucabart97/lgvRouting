#include "lgvRouting/common/CmdParser.h"
#include "lgvRouting/data/Dataset.h"
#include "lgvRouting/heuristic/Heuristic.h"

int main(int argc, char* argv[]) {

    lgv::common::CmdParser cmd(argv, "lgvRouting_launch");
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
    for_each(methods.begin(), methods.end(), [&](const std::string& s){hMethod[s]->init(conf);});

    std::map<std::string, std::pair<int,float>> result;
    int n = d.mNumOfInstances-1;
    for(int i = 0; i < n; i++){
        if(d.loadInstance(problem, i)){
            for_each(methods.begin(), methods.end(), [&](const std::string& s){
                lgv::data::Solution solution = hMethod[s]->run(&problem);
                if (result.find(s) == result.end()){
                    result[s] = std::make_pair(solution.mTime,solution.mCost);
                } else {
                    std::pair<int,float> value = result[s];
                    result[s] = std::make_pair(solution.mTime+value.first,solution.mCost+value.second);
                }
            });
        }
        std::cerr<<"cicle "<<i+1<<"/"<<n<<std::endl;
    }
    std::cout<<"\nAVG results:\n";
    for_each(result.begin(), result.end(), [&](const std::pair<std::string,std::pair<int,float>>& v){
        std::cout<<v.first<<":"<<std::string(30-v.first.size(),' ' )<<v.second.second/n<<"\t"<<v.second.first/n<<"ms"<<std::endl;
    });
    
    lgv::heuristic::dealloc_methods(hMethod);
    return 0;
}