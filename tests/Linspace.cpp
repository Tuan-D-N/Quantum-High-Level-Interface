#include <gtest/gtest.h>
#include "../functionality/Linspace.hpp"

TEST(LinspaceVec, linspaceInclude) {
    std::vector<double> input;
    std::vector<double> output;

    input = linspaceVec(-9.0, -5.4, 10);
    output = { -9,-8.6,-8.2,-7.8,-7.4,-7,-6.6,-6.2,-5.8,-5.4 };

    EXPECT_TRUE(input.size() == output.size());
    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }

    input = linspaceVec(0, 9, 10);
    output = { 0,1,2,3,4,5,6,7,8,9 };
    EXPECT_TRUE(input.size() == output.size());
    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }

    input = linspaceVec(1, 7, 7);
    output = { 1,2,3,4,5,6,7 };
    EXPECT_TRUE(input.size() == output.size());
    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }


    input = linspaceVec(1.0, 4.5, 4);
    output = { 1, 2.166666667, 3.333333333, 4.5 };
    EXPECT_TRUE(input.size() == output.size());

    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }

    input = linspaceVec(1, 6, 7);
    output = { 1,1.83333333333333,2.66666666666667,3.5,4.33333333333333,5.16666666666667,6 };
    EXPECT_TRUE(input.size() == output.size());

    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }
}

TEST(LinspaceVec, linspaceNoInclude) {
    std::vector<double> input;
    std::vector<double> output;


    input = linspaceVec(0, 9, 9, false);
    output = { 0,1,2,3,4,5,6,7,8 };
    EXPECT_TRUE(input.size() == output.size());
    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }

    input = linspaceVec(1, 10, 9, false);
    output = { 1,2,3,4,5,6,7,8,9 };
    EXPECT_TRUE(input.size() == output.size());
    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }

    input = linspaceVec(1.0, 4.5, 7, false);
    output = { 1,1.5,2,2.5,3,3.5,4 };
    EXPECT_TRUE(input.size() == output.size());
    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }

    input = linspaceVec(-10.0, -4.6, 9, false);
    output = { -10,-9.4,-8.8,-8.2,-7.6,-7,-6.4,-5.8,-5.2 };
    EXPECT_TRUE(input.size() == output.size());

    for (int i = 0; i < input.size(); i++)
    {
        EXPECT_NEAR(input[i], output[i], 0.000001);
    }
}