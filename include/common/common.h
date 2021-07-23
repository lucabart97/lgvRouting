#pragma once

#include <yaml-cpp/yaml.h>
#include "common/log.h"

namespace common{

class Entry{
    public:
        int start_node;
        int end_node;
        int priority;
};

class Problem{
    public:
        int n_lgv;
        std::vector<std::vector<int>> cost_matrix;
        std::vector<common::Entry> mission_list;
};

class Mission{
    public:
        int lgv_id;
        Entry entry;
};

class Solution{
    public:
        int best_cost;
        int sol_time;
        std::vector<common::Mission> solution;
};

/**
 * Load YAML node from file
 * @param conf_file
 * @return
 */
inline YAML::Node YAMLloadConf(std::string conf_file) {
    tkDBG("Loading YAML: "<<conf_file<<"\n");
    return YAML::LoadFile(conf_file);
}

/**
 * Get configuration from YAML node
 * @tparam T
 * @param conf yaml node
 * @param key configuration KEY
 * @param defaultVal defalt value in case of no KEY found
 * @return conf value
 */
template<typename T>
inline T YAMLgetConf(YAML::Node conf, std::string key, T defaultVal) {
    T val = defaultVal;
    if(conf && conf[key]) {
        val = conf[key].as<T>();
    }
    //std::cout<<"YAML "<<key<<", val: "<<val<<"\n";
    return val;
}

}