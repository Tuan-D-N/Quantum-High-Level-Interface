#include <iostream>
#include <cuComplex.h>
#include "functionality/WriteAdjMat.hpp"
#include "functionality/GetAdjMat.hpp"
#include "functionality/Utilities.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

int main()
{
    std::string filename = "csvTest/different_types.csv";

    // Create a CSV with integer values
    std::ofstream outFile(filename);
    outFile << "1,2,3\n";
    outFile << "4,5,6\n";
    outFile << "7,8,9\n";
    outFile.close();

    return 0;
}
