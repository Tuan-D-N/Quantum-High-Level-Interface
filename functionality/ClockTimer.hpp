#pragma once
#include <chrono>
#include <iostream>
#include <string>
class Timer
{
private:
    typedef std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> stdTimer;
    std::string name_m = "";
    stdTimer start_m;
    stdTimer stop_m;
    std::chrono::duration<double> duration_m;

public:
    Timer();
    Timer(std::string name);
    void setTime();
    void getTime();
    ~Timer();
};



