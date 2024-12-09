#pragma once
#include <iostream>
#include <string.h>
#include <stdarg.h>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <vector>
#include <list>
#include <chrono>

namespace rtabmap
{

struct Timer {

    Timer(std::string tag = ""){
        tag_ = tag;
        time_start = std::chrono::steady_clock::now();
    }
    ~Timer() {
        auto time_end = std::chrono::steady_clock::now();
        std::cout << tag_<<" use time(ms):" << std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_end - time_start).count() << std::endl;

    }
    void ticks(std::string tmpTag = ""){
        auto time_end = std::chrono::steady_clock::now();
        if(tmpTag.empty()){
            std::cout << tag_ <<" use time(ms):" << std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_end - time_start).count() << std::endl;  
        }else{
            std::cout << tmpTag <<" use time(ms):" << std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_end - time_start).count() << std::endl;  
        }
        time_start = std::chrono::steady_clock::now();
    }
    std::chrono::steady_clock::time_point time_start;
    std::string tag_;
}; 

class UtilsBase
{
public:
    // return the directory path of the file
    static std::string getDir(const std::string & filePath);

    static std::string getName(const std::string & filePath);


    static std::string uFormatv (const char *fmt, va_list args);


    static std::string uFormat (const char *fmt, ...);

    static inline std::list<std::string> uSplit(const std::string & str, char separator = ' ')
    {
        std::list<std::string> v;
        std::string buf;
        for(unsigned int i=0; i<str.size(); ++i)
        {
            if(str[i] != separator)
            {
                buf += str[i];
            }
            else if(buf.size())
            {
                v.push_back(buf);
                buf = "";
            }
        }
        if(buf.size())
        {
            v.push_back(buf);
        }
        return v;
    }

};



}