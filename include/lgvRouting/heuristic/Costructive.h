#pragma once

#include "lgvRouting/heuristic/Generic.h"

namespace lgv { namespace heuristic {


/**
 * @brief   Costructive algorithm. 
 *          Ordering mission by weight and assign it at lgv with minus occupation.
 * 
 */
class Costructive : public Generic{
    public:
        Costructive();
        ~Costructive();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
};

}}