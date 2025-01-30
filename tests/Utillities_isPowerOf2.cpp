#include <gtest/gtest.h>
#include "../functionality/Utilities.hpp"


TEST(IsPowerOf2Test, PowersOf2) {
    EXPECT_TRUE(isPowerOf2(1));
    EXPECT_TRUE(isPowerOf2(2));
    EXPECT_TRUE(isPowerOf2(4));
    EXPECT_TRUE(isPowerOf2(8));
    EXPECT_TRUE(isPowerOf2(16));
    EXPECT_TRUE(isPowerOf2(1024));
}

TEST(IsPowerOf2Test, NonPowersOf2) {
    EXPECT_FALSE(isPowerOf2(0));
    EXPECT_FALSE(isPowerOf2(3));
    EXPECT_FALSE(isPowerOf2(5));
    EXPECT_FALSE(isPowerOf2(10));
    EXPECT_FALSE(isPowerOf2(15));
    EXPECT_FALSE(isPowerOf2(1000));
}

TEST(IsPowerOf2Test, NegativeNumbers) {
    EXPECT_FALSE(isPowerOf2(-1));
    EXPECT_FALSE(isPowerOf2(-2));
    EXPECT_FALSE(isPowerOf2(-4));
    EXPECT_FALSE(isPowerOf2(-8));
}
