#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

template <typename T>
std::vector<std::vector<T>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<T>> data;
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return data;
    }

    // Read each line of the CSV
    while (std::getline(file, line)) {
        std::vector<T> row;
        std::stringstream ss(line);
        std::string cell;

        // Split each line by commas
        while (std::getline(ss, cell, ',')) {
            std::stringstream cellStream(cell);
            T value;
            cellStream >> value;  // Convert string to desired type
            row.push_back(value);
        }

        data.push_back(row);
    }

    file.close();
    return data;
}