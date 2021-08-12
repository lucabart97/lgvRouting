#include <lgvRouting/data/Problem.h>
#include <lgvRouting/data/Dataset.h>

int main() {
    
    lgv::data::Dataset  d;
    d.load(lgv::data::DatasetType::P1);

    lgv::data::Problem  p;
    if (!d.loadInstance(p, 0))
    {
        // do stuff
    }
    
    return 0;
}