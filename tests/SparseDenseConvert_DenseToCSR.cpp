#include <gtest/gtest.h>
#include <vector>
#include <span>
#include <cuComplex.h>
#include "../CuSparseControl/SparseDenseConvert.hpp"



// Small helper
static inline cuDoubleComplex C(double r, double i = 0.0) {
    return make_cuDoubleComplex(r, i);
}

TEST(DenseToCsr, ZeroSizeMatrix) {
    std::vector<cuDoubleComplex> A; // n = 0
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;

    dense_to_csr(std::span<const cuDoubleComplex>(A.data(), 0), 0, row_ptr, col_ind, vals);

    ASSERT_EQ(row_ptr.size(), 1u);
    EXPECT_EQ(row_ptr[0], 0);
    EXPECT_TRUE(col_ind.empty());
    EXPECT_TRUE(vals.empty());
}

TEST(DenseToCsr, AllZeros) {
    const int n = 3;
    std::vector<cuDoubleComplex> A(n*n, C(0.0));
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;

    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals);

    ASSERT_EQ(row_ptr.size(), static_cast<size_t>(n+1));
    EXPECT_EQ(row_ptr, (std::vector<int>{0,0,0,0}));
    EXPECT_TRUE(col_ind.empty());
    EXPECT_TRUE(vals.empty());
}

TEST(DenseToCsr, Identity3) {
    const int n = 3;
    std::vector<cuDoubleComplex> A(n*n, C(0));
    for (int i = 0; i < n; ++i) A[i*n + i] = C(1);

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals);

    EXPECT_EQ(row_ptr, (std::vector<int>{0,1,2,3}));
    EXPECT_EQ(col_ind, (std::vector<int>{0,1,2}));
    ASSERT_EQ(vals.size(), 3u);
    for (auto v : vals) {
        EXPECT_DOUBLE_EQ(cuCreal(v), 1.0);
        EXPECT_DOUBLE_EQ(cuCimag(v), 0.0);
    }
}

TEST(DenseToCsr, GeneralPatternRowMajor) {
    // 3x3
    // [ 1 0 2
    //   0 0 0
    //   3 0 4 ]
    const int n = 3;
    std::vector<cuDoubleComplex> A{
        C(1), C(0), C(2),
        C(0), C(0), C(0),
        C(3), C(0), C(4),
    };

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals);

    EXPECT_EQ(row_ptr, (std::vector<int>{0,2,2,4}));
    EXPECT_EQ(col_ind, (std::vector<int>{0,2,0,2}));

    ASSERT_EQ(vals.size(), 4u);
    EXPECT_DOUBLE_EQ(cuCreal(vals[0]), 1.0); EXPECT_DOUBLE_EQ(cuCimag(vals[0]), 0.0);
    EXPECT_DOUBLE_EQ(cuCreal(vals[1]), 2.0); EXPECT_DOUBLE_EQ(cuCimag(vals[1]), 0.0);
    EXPECT_DOUBLE_EQ(cuCreal(vals[2]), 3.0); EXPECT_DOUBLE_EQ(cuCimag(vals[2]), 0.0);
    EXPECT_DOUBLE_EQ(cuCreal(vals[3]), 4.0); EXPECT_DOUBLE_EQ(cuCimag(vals[3]), 0.0);
}

TEST(DenseToCsr, KeepsColumnOrderWithinRow) {
    // Nonzeros in ascending column order due to j loop
    const int n = 1;
    std::vector<cuDoubleComplex> A{ C(7) };
    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals);
    EXPECT_EQ(col_ind, (std::vector<int>{0}));
    EXPECT_EQ(row_ptr, (std::vector<int>{0,1}));
}

TEST(DenseToCsr, ToleranceDropsNearZeros) {
    // Matrix with tiny entries that should be dropped when tol=1e-8
    const int n = 2;
    std::vector<cuDoubleComplex> A{
        C(1e-12, 5e-12), C(0.0),
        C(0.0),          C(1.0, 0.0)
    };

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, /*tol=*/1e-8);

    EXPECT_EQ(row_ptr, (std::vector<int>{0,0,1}));
    EXPECT_EQ(col_ind, (std::vector<int>{1}));
    ASSERT_EQ(vals.size(), 1u);
    EXPECT_DOUBLE_EQ(cuCreal(vals[0]), 1.0);
    EXPECT_DOUBLE_EQ(cuCimag(vals[0]), 0.0);
}

TEST(DenseToCsr, ComplexPartsCountAsNonZeroIfEitherExceedsTol) {
    // is_zero => both |re| and |im| <= tol. So if imag > tol, it stays.
    const int n = 1;
    std::vector<cuDoubleComplex> A{ C(0.0, 2e-9) };

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals, /*tol=*/1e-9);

    EXPECT_EQ(row_ptr, (std::vector<int>{0,1}));
    EXPECT_EQ(col_ind, (std::vector<int>{0}));
    ASSERT_EQ(vals.size(), 1u);
    EXPECT_DOUBLE_EQ(cuCreal(vals[0]), 0.0);
    EXPECT_DOUBLE_EQ(cuCimag(vals[0]), 2e-9);
}

TEST(DenseToCsr, OutputSizesConsistent) {
    const int n = 4;
    // Diagonal plus one off-diagonal
    std::vector<cuDoubleComplex> A(n*n, C(0));
    for (int i = 0; i < n; ++i) A[i*n + i] = C(1.0);
    A[1*n + 3] = C(-2.0);

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(A), n, row_ptr, col_ind, vals);

    ASSERT_EQ(row_ptr.size(), static_cast<size_t>(n+1));
    ASSERT_EQ(col_ind.size(), vals.size());
    ASSERT_EQ(row_ptr.back(), static_cast<int>(vals.size()));
}
