#include "lgvRouting/common/Time.h"

using namespace lgv::rt;

Time::Time(){

}


Time::~Time(){

}


timeStamp_t 
Time::getTimeStamp() {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return timeStamp_t(tp.tv_sec)*1e6 + tp.tv_nsec/1000;
}


void 
Time::tic() {
    mStart = getTimeStamp();          
}

timeStamp_t 
Time::toc() {
    lgvASSERT(mStart != 0, "must tic() before toc()")
    timeStamp_t res = getTimeStamp() - mStart;
    mStart = 0;
    return res;
}