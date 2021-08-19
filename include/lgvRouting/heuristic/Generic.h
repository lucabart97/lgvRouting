#pragma once

#include "lgvRouting/data/Problem.h"
#include "lgvRouting/data/Solution.h"
#include "lgvRouting/finder/Finder.h"

namespace lgv { namespace heuristic {


/**
 * @brief   Generic algorithm interface
 */
class Generic {
    protected:
        lgv::data::Problem* mProblem = nullptr;
        lgv::data::Solution mSolution;
        lgv::data::Finder   mFinder;
        lgv::rt::Time       mTime;
        timeStamp_t         mTimeout;

    private:
        virtual bool initChild(YAML::Node& aNode) = 0;
        virtual void runChild() = 0;
        virtual bool closeChild() = 0;

    public:
        Generic();
        ~Generic();
        bool init(YAML::Node& aNode);
        lgv::data::Solution run(lgv::data::Problem* aProblem);
        bool close();
};

}}