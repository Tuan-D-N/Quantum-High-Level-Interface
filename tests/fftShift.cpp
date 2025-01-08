#include <gtest/gtest.h>
#include "../functionality/fftShift.hpp"


TEST(fftShift, fftShiftVector1DEven) {
    std::vector<double> data1D = {0, 1, 2, 3, 4, 5, 6, 7};
    fftshift1D(data1D);
    EXPECT_TRUE((data1D == std::vector<double>{4, 5, 6, 7, 0, 1, 2, 3}));

}

TEST(fftShift, fftShiftVector1DOdd) {
    std::vector<double> data1D = {0, 1, 2, 3, 4, 5, 6};
    fftshift1D(data1D);
    EXPECT_TRUE((data1D == std::vector<double>{4, 5, 6, 0, 1, 2, 3}));
}

TEST(fftShift, fftShiftCArray1DEven) {
    double data1D[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int n = sizeof(data1D) / sizeof(data1D[0]);
    fftshift1D(data1D, n);
    double expected1[] = {4, 5, 6, 7, 0, 1, 2, 3};
    for (int i = 0; i < n; ++i) EXPECT_EQ(data1D[i], expected1[i]);

}

TEST(fftShift, fftShiftCArray1DOdd) {
    double data2D[] = {0, 1, 2, 3, 4, 5, 6};
    int n = sizeof(data2D) / sizeof(data2D[0]);
    fftshift1D(data2D, n);
    double expected2[] = {4, 5, 6, 0, 1, 2, 3};
    for (int i = 0; i < n; ++i) EXPECT_EQ(data2D[i], expected2[i]);
}

