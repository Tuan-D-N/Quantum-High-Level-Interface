#include <gtest/gtest.h>
#include "../functionality/ReadCsv.hpp"  // Include the file where readCSV is defined.
#include "../functionality/Utilities.hpp"
#include <vector>
#include <string>

// Test for reading CSV of doubles
TEST(ReadCSVTest, CorrectParsing) {
    std::string filename = "../tests/csvTest/test_data.csv";

    // Create a test CSV file
    std::ofstream outFile(filename);
    outFile << "1.1,2.2,3.3\n";
    outFile << "4.4,5.5,6.6\n";
    outFile << "7.7,8.8,9.9\n";
    outFile.close();

    // Read the CSV file
    std::vector<std::vector<double>> result = readCSV<double>(filename);

    // Expected result
    std::vector<std::vector<double>> expected = {
        {1.1, 2.2, 3.3},
        {4.4, 5.5, 6.6},
        {7.7, 8.8, 9.9}
    };

    // Check if the read data matches the expected data
    EXPECT_TRUE(matricesEqual(result, expected));  // Assuming matricesEqual is defined properly.
}

// Test for edge case: Empty file
TEST(ReadCSVTest, EmptyFile) {
    std::string filename = "../tests/csvTest/empty.csv";

    // Create an empty test CSV file
    std::ofstream outFile(filename);
    outFile.close();

    // Read the empty CSV file
    std::vector<std::vector<double>> result = readCSV<double>(filename);

    // The result should be an empty vector
    EXPECT_TRUE(result.empty());
}

// Test for edge case: Single cell file
TEST(ReadCSVTest, SingleCell) {
    std::string filename = "../tests/csvTest/single_cell.csv";

    // Create a CSV with a single cell
    std::ofstream outFile(filename);
    outFile << "42\n";
    outFile.close();

    // Read the CSV file
    std::vector<std::vector<double>> result = readCSV<double>(filename);

    // Expected result
    std::vector<std::vector<double>> expected = {{42.0}};

    // Check if the result matches the expected data
    EXPECT_TRUE(matricesEqual(result, expected));
}

// Test for edge case: Different types (e.g., integers in the CSV)
TEST(ReadCSVTest, DifferentTypes) {
    std::string filename = "../tests/csvTest/different_types.csv";

    // Create a CSV with integer values
    std::ofstream outFile(filename);
    outFile << "1,2,3\n";
    outFile << "4,5,6\n";
    outFile << "7,8,9\n";
    outFile.close();

    // Read the CSV file as integers
    std::vector<std::vector<int>> result = readCSV<int>(filename);

    // Expected result
    std::vector<std::vector<int>> expected = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Check if the result matches the expected data
    EXPECT_TRUE(matricesEqual(result, expected));
}

