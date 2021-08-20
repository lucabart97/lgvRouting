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
        lgv::data::Problem* mProblem = nullptr; //!< problem pointer
        lgv::data::Solution mSolution;          //!< solution
        lgv::data::Finder   mFinder;            //!< inital solution finder
        lgv::rt::Time       mTime;              //!< timeout utils
        timeStamp_t         mTimeout;           //!< timeout seconds

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