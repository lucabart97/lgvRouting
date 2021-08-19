#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <cmath>

#include "lgvRouting/common/log.h"
#include "lgvRouting/common/Time.h"

typedef int id_lgv;
namespace lgv { namespace common {

/**
 * Load YAML node from file
 * @param conf_file
 * @return
 */
inline YAML::Node YAMLloadConf(std::string conf_file) {
    lgvDBG("Loading YAML: "<<conf_file<<"\n");
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
    return val;
}

}}