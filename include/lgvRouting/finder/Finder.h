#pragma once

#include "lgvRouting/data/Problem.h"
#include "lgvRouting/data/Solution.h"
#include "lgvRouting/common/Time.h"

namespace lgv { namespace data {

    class Finder{
        private:
            lgv::rt::Time mTime;
        public:
            Finder();
            ~Finder();
            /**
             * @brief   Find an initial solution ordering path by cost
             * 
             * @param aProblem  Problem, cost need to be filled
             * @param aLess     Order mission in descresing or incresing order
             * @return Solution Solution
             */
            Solution FindInitialSolution(Problem& aProblem, bool aLess = true);
            /**
             * @brief   Find an random solution
             * 
             * @param aProblem  Problem, cost need to be filled
             * @return Solution Solution
             */
            Solution FindRandomSolution(Problem& aProblem);
            /**
             * @brief   Find an initial solution ordering path with cost and also make the path between
             *          two misson
             * 
             * @param aProblem  Problem, cost need to be filled
             * @return Solution Solution
             */
            Solution FindInitialSolutionWithReturn(Problem& aProblem);
            /**
             * @brief   Fill a solution with return path
             * 
             * @param sol solution that need to be filled
             */
            void FillReturnMission(Solution& sol);
    };
}}