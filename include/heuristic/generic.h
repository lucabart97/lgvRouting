#pragma once

#include "common/common.h"

namespace Heuristic{

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
            tkASSERT(problem != nullptr)
            tkASSERT(sol != nullptr)
            this->sol = sol;
            return this->runChild();
        }

        bool close(){
            return this->closeChild();
        }
};

}