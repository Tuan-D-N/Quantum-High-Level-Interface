#include <gtest/gtest.h>
#include <cuComplex.h>
#include "../functionality/fftShift.hpp"

TEST(fftShiftStride, fftShiftVector1DEven)
{
    std::vector<double> data1D = {0, 1, 2, 3, 4, 5, 6, 7};
    fftshift1D<double>(data1D, 1);
    EXPECT_TRUE((data1D == std::vector<double>{4, 5, 6, 7, 0, 1, 2, 3}));
}

// TEST(fftShiftStride, fftShiftVector1DOdd)
// {
//     std::vector<double> data1D = {0, 1, 2, 3, 4, 5, 6};
//     fftshift1D<double>(data1D, 1);
//     EXPECT_TRUE((data1D == std::vector<double>{4, 5, 6, 0, 1, 2, 3}));
// }

TEST(fftShiftStride, fftShiftCArray1DEven)
{
    double data1D[] = {0, 1, 2, 3, 4, 5, 6, 7};
    int n = sizeof(data1D) / sizeof(data1D[0]);
    fftshift1D(data1D, n, 1);
    double expected1[] = {4, 5, 6, 7, 0, 1, 2, 3};
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(data1D[i], expected1[i]);
}

// TEST(fftShiftStride, fftShiftCArray1DOdd)
// {
//     double data2D[] = {0, 1, 2, 3, 4, 5, 6};
//     int n = sizeof(data2D) / sizeof(data2D[0]);
//     fftshift1D(data2D, n, 1);
//     double expected2[] = {4, 5, 6, 0, 1, 2, 3};
//     for (int i = 0; i < n; ++i)
//         EXPECT_EQ(data2D[i], expected2[i]);
// }

TEST(fftShiftStride, fftShiftCuDoubleComplex1DEven)
{
    cuDoubleComplex data1D[] = {
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 1.0),
        make_cuDoubleComplex(2.0, 2.0), make_cuDoubleComplex(3.0, 3.0),
        make_cuDoubleComplex(4.0, 4.0), make_cuDoubleComplex(5.0, 5.0),
        make_cuDoubleComplex(6.0, 6.0), make_cuDoubleComplex(7.0, 7.0)};
    int n = sizeof(data1D) / sizeof(data1D[0]);
    fftshift1D(data1D, n, 1);
    cuDoubleComplex expected1[] = {
        make_cuDoubleComplex(4.0, 4.0), make_cuDoubleComplex(5.0, 5.0),
        make_cuDoubleComplex(6.0, 6.0), make_cuDoubleComplex(7.0, 7.0),
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 1.0),
        make_cuDoubleComplex(2.0, 2.0), make_cuDoubleComplex(3.0, 3.0)};
    for (int i = 0; i < n; ++i)
    {
        EXPECT_EQ(cuCreal(data1D[i]), cuCreal(expected1[i]));
        EXPECT_EQ(cuCimag(data1D[i]), cuCimag(expected1[i]));
    }
}

// TEST(fftShiftStride, fftShiftCuDoubleComplex1DOdd)
// {
//     cuDoubleComplex data2D[] = {
//         make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 1.0),
//         make_cuDoubleComplex(2.0, 2.0), make_cuDoubleComplex(3.0, 3.0),
//         make_cuDoubleComplex(4.0, 4.0), make_cuDoubleComplex(5.0, 5.0),
//         make_cuDoubleComplex(6.0, 6.0)};
//     int n = sizeof(data2D) / sizeof(data2D[0]);
//     fftshift1D(data2D, n, 1);
//     cuDoubleComplex expected2[] = {
//         make_cuDoubleComplex(4.0, 4.0), make_cuDoubleComplex(5.0, 5.0),
//         make_cuDoubleComplex(6.0, 6.0), make_cuDoubleComplex(0.0, 0.0),
//         make_cuDoubleComplex(1.0, 1.0), make_cuDoubleComplex(2.0, 2.0),
//         make_cuDoubleComplex(3.0, 3.0)};
//     for (int i = 0; i < n; ++i)
//     {
//         EXPECT_EQ(cuCreal(data2D[i]), cuCreal(expected2[i]));
//         EXPECT_EQ(cuCimag(data2D[i]), cuCimag(expected2[i]));
//     }
// }

