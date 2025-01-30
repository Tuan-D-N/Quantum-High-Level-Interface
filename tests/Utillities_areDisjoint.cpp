#include <gtest/gtest.h>
#include <span>
#include <initializer_list>
#include <vector>
#include "../functionality/Utilities.hpp" // Include the header file for your function

// Test case when there is no overlap
TEST(AreDisjointTest, NoOverlap) {
    std::initializer_list<const int> targets = {1, 2, 3};
    std::initializer_list<const int> controls = {4, 5, 6};
    
    EXPECT_TRUE(are_disjoint(targets, controls));  // No overlap
}

// Test case when there is an overlap
TEST(AreDisjointTest, Overlap) {
    std::initializer_list<const int> targets = {1, 2, 3};
    std::initializer_list<const int> controls = {2, 4, 5};
    
    EXPECT_FALSE(are_disjoint(targets, controls));  // Overlap found (2)
}

// Test case when all elements overlap
TEST(AreDisjointTest, AllOverlap) {
    std::initializer_list<const int> targets = {1, 2, 3};
    std::initializer_list<const int> controls = {1, 2, 3};
    
    EXPECT_FALSE(are_disjoint(targets, controls));  // All elements overlap
}

// Test case with repeated elements in both spans
TEST(AreDisjointTest, RepeatedElements) {
    std::initializer_list<const int> targets = {1, 2, 2, 3};
    std::initializer_list<const int> controls = {2, 2, 3, 4};
    
    EXPECT_FALSE(are_disjoint(targets, controls));  // Overlap found (2, 3)
}

// Test case with empty spans
TEST(AreDisjointTest, EmptySpans) {
    std::initializer_list<const int> targets = {};
    std::initializer_list<const int> controls = {};
    
    EXPECT_TRUE(are_disjoint(targets, controls));  // No overlap (both empty)
}

// Test case where one span is empty
TEST(AreDisjointTest, OneEmptySpan) {
    std::initializer_list<const int> targets = {1, 2, 3};
    std::initializer_list<const int> controls = {};
    
    EXPECT_TRUE(are_disjoint(targets, controls));  // No overlap (one empty)
}
