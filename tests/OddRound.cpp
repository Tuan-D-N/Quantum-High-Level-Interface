
#include <gtest/gtest.h>
#include "../functionality/OddRound.hpp"
#include <vector>

TEST(OddRounding, roundUp) {
  EXPECT_EQ(roundToHigherOdd(0)    , 1);
  EXPECT_EQ(roundToHigherOdd(0.25) , 1);
  EXPECT_EQ(roundToHigherOdd(0.5)  , 1);
  EXPECT_EQ(roundToHigherOdd(1)    , 1);
  EXPECT_EQ(roundToHigherOdd(2)    , 3);
  EXPECT_EQ(roundToHigherOdd(-1)   , -1);
  EXPECT_EQ(roundToHigherOdd(-0.5) , 1);
  EXPECT_EQ(roundToHigherOdd(-0.25), 1);
  EXPECT_EQ(roundToHigherOdd(-2)   , -1);
}

TEST(OddRounding, roundDown) {
  EXPECT_EQ(roundToLowerOdd(0)     , -1);
  EXPECT_EQ(roundToLowerOdd(0.25)  , -1);
  EXPECT_EQ(roundToLowerOdd(0.5)   , -1);
  EXPECT_EQ(roundToLowerOdd(1)     , 1);
  EXPECT_EQ(roundToLowerOdd(2)     , 1);
  EXPECT_EQ(roundToLowerOdd(-1)    , -1);
  EXPECT_EQ(roundToLowerOdd(-0.5)  , -1);
  EXPECT_EQ(roundToLowerOdd(-0.25) , -1);
  EXPECT_EQ(roundToLowerOdd(-2)    , -3);
  EXPECT_EQ(roundToLowerOdd(-3)    , -3);
}


