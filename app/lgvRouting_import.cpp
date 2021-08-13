#include <lgvRouting/data/Problem.h>
#include <lgvRouting/data/Dataset.h>

int main() {
    
    lgv::data::Dataset  d;
    d.load(lgv::data::DatasetType::P2);

    lgv::data::Problem  p;
    if (d.loadInstance(p, 0))
    {
        // do stuff
        std::cout<<"Caricato problema con "<<p.mMissions.size()<<" missioni.\n";
    }
    
    return 0;
}