#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <map>
#include <thread>
#include <mutex>
#include "lgvRouting/common/common.h"

typedef uint64_t timeStamp_t;

namespace lgv { namespace rt {

class Time{
    private:
        timeStamp_t mStart = 0;
    public:
        Time();
        ~Time();
        void tic();
        timeStamp_t toc();
    private:
        timeStamp_t getTimeStamp();
};

}}