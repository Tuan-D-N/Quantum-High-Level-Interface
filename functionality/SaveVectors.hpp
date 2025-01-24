#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

// Template function to save a vector of vectors of any type
template <typename T>
void save_vector_of_vectors(const std::vector<std::vector<T>> &vec, const std::string &filename)
{
    std::ofstream out_file(filename, std::ios::binary); // Open file in binary mode
    if (!out_file)
    {
        throw std::runtime_error("Failed to open file for writing!");
    }

    // Write the size of the outer vector (number of inner vectors)
    size_t outer_size = vec.size();
    out_file.write(reinterpret_cast<const char *>(&outer_size), sizeof(outer_size));

    // For each inner vector, write its size and contents
    for (const auto &inner_vec : vec)
    {
        size_t inner_size = inner_vec.size();
        out_file.write(reinterpret_cast<const char *>(&inner_size), sizeof(inner_size));
        out_file.write(reinterpret_cast<const char *>(inner_vec.data()), inner_size * sizeof(T));
    }

    out_file.close();
    std::cout << "Vector of vectors saved to " << filename << std::endl;
}

// Template function to load a vector of vectors of any type
template <typename T>
std::vector<std::vector<T>> load_vector_of_vectors(const std::string &filename)
{
    std::ifstream in_file(filename, std::ios::binary); // Open file in binary mode
    if (!in_file)
    {
        throw std::runtime_error("Failed to open file for reading!");
    }

    // Read the size of the outer vector (number of inner vectors)
    size_t outer_size;
    in_file.read(reinterpret_cast<char *>(&outer_size), sizeof(outer_size));

    // Create the outer vector
    std::vector<std::vector<T>> vec(outer_size);

    // For each inner vector, read its size and its contents
    for (auto &inner_vec : vec)
    {
        size_t inner_size;
        in_file.read(reinterpret_cast<char *>(&inner_size), sizeof(inner_size));
        inner_vec.resize(inner_size); // Resize the inner vector to the correct size
        in_file.read(reinterpret_cast<char *>(inner_vec.data()), inner_size * sizeof(T));
    }

    in_file.close();
    std::cout << "Vector of vectors loaded from " << filename << std::endl;

    return vec;
}

// Template function to save a vector of any type
template <typename T>
void save_vector(const std::vector<T> &vec, const std::string &filename)
{
    std::ofstream out_file(filename, std::ios::binary); // Open file in binary mode
    if (!out_file)
    {
        throw std::runtime_error("Failed to open file for writing!");
    }

    // Write the size of the vector
    size_t size = vec.size();
    out_file.write(reinterpret_cast<const char *>(&size), sizeof(size));

    // Write the elements of the vector
    out_file.write(reinterpret_cast<const char *>(vec.data()), size * sizeof(T));

    out_file.close();
    std::cout << "Vector saved to " << filename << std::endl;
}

// Template function to load a vector of any type
template <typename T>
std::vector<T> load_vector(const std::string &filename)
{
    std::ifstream in_file(filename, std::ios::binary); // Open file in binary mode
    if (!in_file)
    {
        throw std::runtime_error("Failed to open file for reading!");
    }

    // Read the size of the vector
    size_t size;
    in_file.read(reinterpret_cast<char *>(&size), sizeof(size));

    // Create a vector of the appropriate size
    std::vector<T> vec(size);

    // Read the elements of the vector
    in_file.read(reinterpret_cast<char *>(vec.data()), size * sizeof(T));

    in_file.close();
    std::cout << "Vector loaded from " << filename << std::endl;

    return vec;
}
