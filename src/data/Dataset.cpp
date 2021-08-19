#include "lgvRouting/data/Dataset.h"

using namespace lgv::data;

Dataset::Dataset(){
}

Dataset::~Dataset(){
}

bool 
Dataset::fileExist(const char *fname){
    std::ifstream dataFile (fname, std::ios::in | std::ios::binary);
    if(!dataFile)
        return false;
    return true;
}

bool 
Dataset::downloadDatasetIfDoNotExist(const std::string& input_bin, const std::string& dataset_folder, const std::string& dataset_url){
    if(!fileExist(input_bin.c_str())){
        std::string mkdir_cmd   = "mkdir " + dataset_folder; 
        std::string wget_cmd    = "curl " + dataset_url + " --output " + dataset_folder + "/dataset.zip";
        std::string unzip_cmd   = "unzip " + dataset_folder + "/dataset.zip -d" + dataset_folder;
        std::string rm_cmd      = "rm " + dataset_folder + "/dataset.zip";

        int err = 0;
        err = system(mkdir_cmd.c_str());
        err = system(wget_cmd.c_str());
        err = system(unzip_cmd.c_str());
        err = system(rm_cmd.c_str());

        if (err == 0)
            return true;
        else 
            return false;
    } 
    return true;
}

bool
Dataset::load(const DatasetType aType) {
    std::string input_bins, curl_path;
    switch (aType)
    {
    case DatasetType::P1:
        mBinPath    = "P1";
        input_bins  = "/README.P1";
        curl_path   = "https://cloud.hipert.unimore.it/s/53HaWSDKsa3kKNW/download";
        mType       = aType;
        break;
    
    case DatasetType::P2:
        mBinPath    = "P2";
        input_bins  = "/README.P2";
        curl_path   = "https://cloud.hipert.unimore.it/s/gempDFcdCiBz4g5/download";
        mType       = aType;
        break;
    
    default:
        lgvERR("Dataset type not supported.")
        return false;
    }
    
    // download dataset 
    if (!downloadDatasetIfDoNotExist(mBinPath + input_bins, mBinPath, curl_path)) {
        lgvERR("Error while downloading dataset '"<<mBinPath<<"'.");
        return false;
    }

    // read readme
    std::fstream newfile;
    newfile.open(mBinPath + input_bins, std::ios::in); 
    if (newfile.is_open()) {  
        std::string tp;
        int line_id = 0;
        while (getline(newfile, tp)) { 
            //std::cout << "["<<line_id<<"] "<<tp << "\n";   //print the data of the string
            switch (line_id)
            {
            case 0:
                mName           = tp;
                break;
            
            case 2:
                mNumOfInstances = std::atoi(tp.substr(0, tp.find_first_of(' ')).c_str());
                break;
            
            case 3:
                mNumOfDepos     = std::atoi(tp.substr(tp.find_first_of(':') + 2, tp.length()).c_str());
                break;

            case 4:
                mNumOfStops     = std::atoi(tp.substr(tp.find_first_of(':') + 2, tp.length()).c_str());
                break;
            
            case 8:
            {
                std::string tmp     = tp.substr(tp.find_first_of(':') + 1, tp.length());
                mNumOfVehicles = 10;
            }
                break;
            
            case 9:
                mCapacity       = std::atoi(tp.substr(tp.find_first_of('=') + 1, tp.length()).c_str());
                break;
            
            default:
                break;
            }
            line_id++;
        }
        newfile.close();   //close the file object.
    }

    /*std::cout<<"Loaded "<<mName<<" dataset:"<< 
               "\nVRP instances:\t\t"<<mNumOfInstances<<
               "\nNr. of Depots:\t\t"<<mNumOfDepos<<
               "\nNr. of Stops:\t\t"<<mNumOfStops<<
               "\nCapacity of vehicles:\t"<<mCapacity<<"\n";*/

    return true;
}

bool
Dataset::loadInstance(Problem& aProblem, const uint32_t aIdx){
    if ((aIdx + 1) < mNumOfInstances) {
        std::fstream newfile;
        newfile.open(mBinPath + "/" + std::to_string(aIdx + 1) + mBinPath + ".DAT", std::ios::in);
        if (newfile.is_open()) {  
            std::string tp;
            int id, x, y, open_tw1, close_tw1, open_tw2, close_tw2, demand, service_time, type;
            while (getline(newfile, tp)) { 
                // decode string
                sscanf(tp.c_str(), "%d %d %d %d %d %d %d %d %d %d", &id, &x, &y, &open_tw1, &close_tw1, &open_tw2, &close_tw2, &demand, &service_time, &type);
                
                Location l(id, float(x), float(y), demand, open_tw1, close_tw1, open_tw2, close_tw2, service_time);

                if (id == 0) {
                    aProblem.mDepot = l;
                    continue;
                }
                
                switch (mType)
                {
                case DatasetType::P1:
                {
                    if (type == 0)
                        aProblem.mDelivery.push_back(l);
                    else 
                        aProblem.mPickUp.push_back(l);
                }
                    break;
                case DatasetType::P2:
                {
                    if (id % 2 == 0)
                        aProblem.mDelivery.push_back(l);
                    else 
                        aProblem.mPickUp.push_back(l);
                }
                    break;
                default:
                    break;
                }
            }
            newfile.close();

            for (int i = 0; i < aProblem.mPickUp.size(); ++i) {
                if (i < aProblem.mDelivery.size()) 
                    aProblem.mMissions.push_back(lgv::data::Mission(aProblem.mPickUp[i], aProblem.mDelivery[i], 0));
                else 
                    break;
            }

            aProblem.mCapacity              = mCapacity;
            aProblem.mNumberOfVeichles      = mNumOfVehicles;
        } else {
            lgvERR("Cannot open file.");
            return false;
        }
    } else {
        lgvWRN("");
        return false;
    }
    return true;
}