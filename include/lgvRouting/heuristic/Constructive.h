#pragma once

#include "lgvRouting/heuristic/Generic.h"

namespace lgv { namespace heuristic {


/**
 * @brief   Constructive algorithm. 
 *          Ordering mission by weight and assign it at lgv with minus occupation.
 * 
 */
class Constructive : public Generic{
    private:
        bool mDecreasing;
    public:
        Constructive();
        ~Constructive();
    private:
        bool initChild(YAML::Node& aNode) override;
        void runChild() override;
        bool closeChild() override;
};

}}