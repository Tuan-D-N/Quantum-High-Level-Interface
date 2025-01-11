#include "ClockTimer.hpp"

Timer::Timer()
{
    start_m = std::chrono::high_resolution_clock::now();
}
Timer::Timer(std::string name) : name_m(name)
{
    start_m = std::chrono::high_resolution_clock::now();
}

Timer::~Timer()
{
    stop_m = std::chrono::high_resolution_clock::now();
    duration_m = std::chrono::duration<double>(stop_m - start_m);
    std::cout << name_m << "," << duration_m.count() << std::endl;
}

