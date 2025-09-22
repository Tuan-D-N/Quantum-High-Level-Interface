#pragma once
#include <chrono>
#include <iostream>
#include <string>
/**
 * @class Timer
 * @brief A simple timer class for measuring elapsed time using std::chrono.
 */
class Timer
{
private:
    /// Alias for std::chrono system clock time point.
    typedef std::chrono::time_point<std::chrono::_V2::system_clock, std::chrono::_V2::system_clock::duration> stdTimer;

    std::string name_m = "";                   ///< Optional name for the timer.
    stdTimer start_m;                          ///< Start time point.
    stdTimer stop_m;                           ///< Stop time point.
    std::chrono::duration<double> duration_m;  ///< Measured duration.

public:
    /**
     * @brief Default constructor. Initializes the timer without a name.
     */
    Timer();

    /**
     * @brief Constructor with name.
     * @param name A string to label the timer.
     */
    Timer(std::string name);

    /**
     * @brief Records the current time as the start time.
     */
    void setTime();

    /**
     * @brief Records the current time as the stop time and computes the duration.
     */
    void getTime();

    /**
     * @brief Destructor.
     */
    ~Timer();
};




