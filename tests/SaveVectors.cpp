#include <gtest/gtest.h>
#include "../functionality/SaveVectors.hpp" // Include the header file with the template functions
#include <vector>
#include <fstream>
#include <string>

#include <filesystem> // Requires C++17 or later

// Helper function to check and remove a file if it exists
void remove_file(const std::string &filename, const bool shouldExist)
{

    if (std::filesystem::exists(filename) ^ shouldExist) // Mismatch case
    {
        std::cerr << "File Name: " << filename << "\n";
        std::cerr << "shouldExist: " << shouldExist << "\n";
        throw std::runtime_error("Mismatch case of file and should Exist");
    }

    std::remove(filename.c_str());
}

// Test saving and loading a vector of integers
TEST(VectorIO, SaveAndLoadVectorInt)
{
    std::string filename = "test_vector_int.bin";
    std::vector<int> original = {1, 2, 3, 4, 5};
    remove_file(filename, false);

    // Save and load
    save_vector(original, filename);
    std::vector<int> loaded = load_vector<int>(filename);

    // Assertions
    ASSERT_EQ(original.size(), loaded.size());
    ASSERT_EQ(original, loaded);

    remove_file(filename, true);
}

// Test saving and loading a vector of doubles
TEST(VectorIO, SaveAndLoadVectorDouble)
{
    std::string filename = "test_vector_double.bin";
    std::vector<double> original = {1.1, 2.2, 3.3};
    remove_file(filename, false);

    save_vector(original, filename);
    std::vector<double> loaded = load_vector<double>(filename);

    ASSERT_EQ(original.size(), loaded.size());
    ASSERT_EQ(original, loaded);

    remove_file(filename, true);
}

// Test saving and loading a vector of vectors of integers
TEST(VectorIO, SaveAndLoadVectorOfVectorsInt)
{
    std::string filename = "test_vector_of_vectors_int.bin";
    std::vector<std::vector<int>> original = {{1, 2}, {3, 4, 5}, {6}};
    remove_file(filename, false);

    save_vector_of_vectors(original, filename);
    std::vector<std::vector<int>> loaded = load_vector_of_vectors<int>(filename);

    ASSERT_EQ(original.size(), loaded.size());
    ASSERT_EQ(original, loaded);

    remove_file(filename, true);
}

// Test saving and loading a vector of vectors of doubles
TEST(VectorIO, SaveAndLoadVectorOfVectorsDouble)
{
    std::string filename = "test_vector_of_vectors_double.bin";
    std::vector<std::vector<double>> original = {{1.1, 2.2}, {3.3}, {4.4, 5.5, 6.6}};
    remove_file(filename, false);

    save_vector_of_vectors(original, filename);
    std::vector<std::vector<double>> loaded = load_vector_of_vectors<double>(filename);

    ASSERT_EQ(original.size(), loaded.size());
    ASSERT_EQ(original, loaded);

    remove_file(filename, true);
}

// Test file error handling for saving
TEST(VectorIO, SaveFileError)
{
    std::string filename = "/invalid_path/test_file_error.bin";
    std::vector<int> vec = {1, 2, 3};

    EXPECT_THROW(save_vector(vec, filename), std::runtime_error);
}

// Test file error handling for loading
TEST(VectorIO, LoadFileError)
{
    std::string filename = "non_existent_file.bin";

    EXPECT_THROW(load_vector<int>(filename), std::runtime_error);
}

// Test empty vector saving and loading
TEST(VectorIO, SaveAndLoadEmptyVector)
{
    std::string filename = "test_empty_vector.bin";
    std::vector<int> original;
    remove_file(filename, false);

    save_vector(original, filename);
    std::vector<int> loaded = load_vector<int>(filename);

    ASSERT_TRUE(loaded.empty());
    remove_file(filename, true);
}

// Test empty vector of vectors saving and loading
TEST(VectorIO, SaveAndLoadEmptyVectorOfVectors)
{
    std::string filename = "test_empty_vector_of_vectors.bin";
    std::vector<std::vector<int>> original;
    remove_file(filename, false);

    save_vector_of_vectors(original, filename);
    std::vector<std::vector<int>> loaded = load_vector_of_vectors<int>(filename);

    ASSERT_TRUE(loaded.empty());
    remove_file(filename, true);
}
