#pragma once

#include <iostream>
#include <fstream>
#include <string>

#include "lgvRouting/data/Problem.h"

namespace lgv { namespace data {

enum class DatasetType {
    P1,
    P2
};

class Dataset
{
public:
    uint32_t    mNumOfInstances;
private:
    uint32_t    mNumOfDepos;
    uint32_t    mNumOfStops;
    uint32_t    mNumOfVehicles;
    uint32_t    mCapacity;
    std::string mName;
    std::string mBinPath;
    DatasetType mType;

    bool fileExist(const char *fname);
    bool downloadDatasetIfDoNotExist(const std::string& input_bin, const std::string& dataset_folder, const std::string& dataset_url);
public:
     Dataset(/* args */);
    ~Dataset();

    bool load(const DatasetType aType);
    bool loadInstance(Problem& aProblem, const uint32_t aIdx);
};

}}