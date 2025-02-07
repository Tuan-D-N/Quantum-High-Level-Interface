#include "ClockTimer.hpp"

Timer::Timer()
{
    setTime();
}
Timer::Timer(std::string name) : Timer()
{
    name_m = std::move(name); // Initialize name_m in the body
}

void Timer::setTime()
{
    start_m = std::chrono::high_resolution_clock::now();
}

void Timer::getTime()
{
    stop_m = std::chrono::high_resolution_clock::now();
    duration_m = std::chrono::duration<double>(stop_m - start_m);
    std::cout << name_m << ", Time = " << duration_m.count() << std::endl;
}

Timer::~Timer()
{
    getTime();
}
