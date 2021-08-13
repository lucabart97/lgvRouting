#pragma once

//#include "lgvRouting/common/common.h"

namespace lgv { namespace heuristic {

/*
class Generic {
    private:
        common::Problem* problem = nullptr;
        common::Solution* sol    = nullptr;

        virtual bool initChild(YAML::Node& node);
        virtual bool runChild();
        virtual bool closeChild();

    public:
        bool init(YAML::Node& node, common::Problem* problem){
            this->problem = problem;
            return this->initChild(node);
        }

        bool run(common::Solution* sol){
            lgvASSERT(problem != nullptr)
            lgvASSERT(sol != nullptr)
            this->sol = sol;
            return this->runChild();
        }

        bool close(){
            return this->closeChild();
        }
};
*/

}}