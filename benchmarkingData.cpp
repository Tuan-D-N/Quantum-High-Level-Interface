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
}

void time_execution(int nQubits)
{
    const auto stateVec = generateNormalizedRandomVectorState<cuComplex>(nQubits);
    quantumState_SV<precision::bit_32> state(stateVec);
    {
        std::ostringstream name;
        name << "number of qubits: " << nQubits;
        Timer(name.str());
        
        circuitExecution(state, nQubits);
    }
}

int main(int argc, char const *argv[])
{
    std::cout << "begin" << std::endl;
    for(int i = 1; i < 30; ++i)
    {
        time_execution(i);
    }
    std::cout << "end" << std::endl;
    return 0;
}
