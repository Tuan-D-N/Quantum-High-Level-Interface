#include <gtest/gtest.h>
#include "../functionality/Utilities.hpp"

TEST(Utilities, isOdd) {
    std::vector<int> oddNumbers{ -21,-9,-7,-5,-3,-1,1,3,5,7,9,11 };
    for (auto number : oddNumbers)
    {
        EXPECT_TRUE(isOdd(number));
    }

    std::vector<int> evenNumbers{ -20,-18,-14,-12,-6,-4,-2,0,2,4,6,8,10,30,50 };
    for (auto number : evenNumbers)
    {
        EXPECT_FALSE(isOdd(number));
    }

}

TEST(Utilities, isEven) {
    std::vector<int> oddNumbers{ -21,-9,-7,-5,-3,-1,1,3,5,7,9,11 };
    for (auto number : oddNumbers)
    {
        EXPECT_FALSE(isEven(number));
    }

    std::vector<int> evenNumbers{ -20,-18,-14,-12,-6,-4,-2,0,2,4,6,8,10,30,50 };
    for (auto number : evenNumbers)
    {
        EXPECT_TRUE(isEven(number));
    }

}