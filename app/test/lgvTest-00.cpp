#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <lgvRouting/catch2/catch.hpp>

#include "lgvRouting/data/Dataset.h"
#include "lgvRouting/heuristic/Costructive.h"

float truncate(float val){
  return floor(val*1000.)/1000;
}

TEST_CASE("Test heuristic costructive") {
    lgv::data::Dataset  d;
    d.load(lgv::data::DatasetType::P2);

    lgv::data::Problem problem;
    lgv::heuristic::Costructive c;

    d.loadInstance(problem, 0);
    REQUIRE(truncate(c.run(&problem).mCost) == 277.01401f);
}