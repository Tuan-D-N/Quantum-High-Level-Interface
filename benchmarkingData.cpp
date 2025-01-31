#include <iostream>
#include <vector>
#include <array>
#include "CuQuantumControl/StateObject.hpp"
#include "functionality/ClockTimer.hpp"
#include "functionality/randomArray.hpp"
#include <sstream>

void circuitExecution(quantumState_SV<precision::bit_32> &state, int nQubits)
{
    for (int block = 0; block < 3; ++block)
    {
        for (int i = 0; i < nQubits; ++i)
        {
            state.RY(i * 10, {i});
            if (i > 0)
            {
                state.X({i - 1}, {i});
            }
        }
    }
    cudaDeviceSynchronize();
}


void time_execution(int nQubits)
{
    std::ostringstream name;
    name << "Total Creation and Run. number of qubits: " << nQubits;
    auto a = Timer(name.str());
        
    quantumState_SV<precision::bit_32> state(nQubits);
    generateNormalizedRandomStateWrite<cuComplex>(state.getStateVector());
    {
        std::ostringstream name;
        name << "Just circuit run time. number of qubits: " << nQubits;
        auto b = Timer(name.str());
        
        circuitExecution(state, nQubits);
    }
}

int main(int argc, char const *argv[])
{
    std::cout << "begin" << std::endl;
    for(int i = 25; i < 30; ++i)
    {
        time_execution(i);
    }
    std::cout << "end" << std::endl;
    return 0;
}
