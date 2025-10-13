#include <gtest/gtest.h>
#include <vector>
#include <span>
#include <cuComplex.h>
#include "../CuSparseControl/SparseDenseConvert.hpp"


// #include "csr_to_dense.hpp"  // your header with csr_to_dense, zero_cuDoubleComplex

static inline cuDoubleComplex C(double r, double i = 0.0) {
    return make_cuDoubleComplex(r, i);
}

static inline bool CEq(cuDoubleComplex a, cuDoubleComplex b, double eps=0.0) {
    return std::abs(cuCreal(a)-cuCreal(b)) <= eps && std::abs(cuCimag(a)-cuCimag(b)) <= eps;
}

TEST(CsrToDense, ZeroSizeMatrix) {
    std::vector<int> row_ptr{0};
    std::vector<int> col_ind;
    std::vector<cuDoubleComplex> vals;
    std::vector<cuDoubleComplex> A;

    csr_to_dense(std::span<const int>(row_ptr),
                 std::span<const int>(col_ind),
                 std::span<const cuDoubleComplex>(vals),
                 /*n=*/0, A);

    EXPECT_TRUE(A.empty());
}

TEST(CsrToDense, AllZeros3x3) {
    const int n = 3;
    std::vector<int> row_ptr{0,0,0,0};
    std::vector<int> col_ind;
    std::vector<cuDoubleComplex> vals;
    std::vector<cuDoubleComplex> A;

    csr_to_dense(row_ptr, col_ind, vals, n, A);

    ASSERT_EQ(A.size(), static_cast<size_t>(n*n));
    for (auto z : A) {
        EXPECT_TRUE(CEq(z, zero_cuDoubleComplex()));
    }
}

TEST(CsrToDense, Identity3) {
    const int n = 3;
    // CSR for I_3
    std::vector<int> row_ptr{0,1,2,3};
    std::vector<int> col_ind{0,1,2};
    std::vector<cuDoubleComplex> vals{C(1),C(1),C(1)};
    std::vector<cuDoubleComplex> A;

    csr_to_dense(row_ptr, col_ind, vals, n, A);

    ASSERT_EQ(A.size(), 9u);
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            cuDoubleComplex want = (i==j)? C(1): C(0);
            EXPECT_TRUE(CEq(A[i*n + j], want)) << "i="<<i<<" j="<<j;
        }
    }
}

TEST(CsrToDense, GeneralPatternRowMajor) {
    // CSR of:
    // [ 1 0 2
    //   0 0 0
    //   3 0 4 ]
    const int n = 3;
    std::vector<int> row_ptr{0,2,2,4};
    std::vector<int> col_ind{0,2,0,2};
    std::vector<cuDoubleComplex> vals{C(1),C(2),C(3),C(4)};
    std::vector<cuDoubleComplex> A;

    csr_to_dense(row_ptr, col_ind, vals, n, A);

    ASSERT_EQ(A.size(), 9u);
    EXPECT_TRUE(CEq(A[0*n + 0], C(1)));
    EXPECT_TRUE(CEq(A[0*n + 2], C(2)));
    EXPECT_TRUE(CEq(A[2*n + 0], C(3)));
    EXPECT_TRUE(CEq(A[2*n + 2], C(4)));

    // All other entries should be zero
    for (int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            if ((i==0 && (j==0||j==2)) || (i==2 && (j==0||j==2))) continue;
            EXPECT_TRUE(CEq(A[i*n + j], C(0))) << "Nonzero where zero expected at ("<<i<<","<<j<<")";
        }
    }
}

TEST(CsrToDense, ComplexValuesPreserved) {
    const int n = 2;
    // [ 0    2-3i
    //   1+4i 0   ]
    std::vector<int> row_ptr{0,1,2};
    std::vector<int> col_ind{1,0};
    std::vector<cuDoubleComplex> vals{C(2,-3), C(1,4)};
    std::vector<cuDoubleComplex> A;

    csr_to_dense(row_ptr, col_ind, vals, n, A);

    EXPECT_TRUE(CEq(A[0*n + 0], C(0,0)));
    EXPECT_TRUE(CEq(A[0*n + 1], C(2,-3)));
    EXPECT_TRUE(CEq(A[1*n + 0], C(1,4)));
    EXPECT_TRUE(CEq(A[1*n + 1], C(0,0)));
}

TEST(CsrToDense, HandlesEmptyMiddleRow) {
    // 4x4 with row 2 empty
    const int n = 4;
    // Nonzeros at (0,1)=5, (2,3)=7
    std::vector<int> row_ptr{0,1,1,2,2};
    std::vector<int> col_ind{1,3};
    std::vector<cuDoubleComplex> vals{C(5),C(7)};
    std::vector<cuDoubleComplex> A;

    csr_to_dense(row_ptr, col_ind, vals, n, A);

    ASSERT_EQ(A.size(), 16u);
    EXPECT_TRUE(CEq(A[0*n + 1], C(5)));
    EXPECT_TRUE(CEq(A[2*n + 3], C(7)));
    // Check an empty row stayed zero
    for (int j=0;j<n;j++) {
        EXPECT_TRUE(CEq(A[1*n + j], C(0))) << "Row 1 not zero at col " << j;
    }
}

#if GTEST_HAS_DEATH_TEST
TEST(CsrToDenseDeath, AssertsOnRowPtrSizeMismatch) {
    const int n = 3;
    std::vector<int> bad_row_ptr{0,1,2}; // should be size n+1=4
    std::vector<int> col_ind{0,1};
    std::vector<cuDoubleComplex> vals{C(1),C(1)};
    std::vector<cuDoubleComplex> A;

    EXPECT_DEATH(
        csr_to_dense(bad_row_ptr, col_ind, vals, n, A),
        ""
    );
}

TEST(CsrToDenseDeath, AssertsOnColIndOutOfRange) {
    const int n = 2;
    std::vector<int> row_ptr{0,1,1};
    std::vector<int> col_ind{5}; // invalid (>= n)
    std::vector<cuDoubleComplex> vals{C(1)};
    std::vector<cuDoubleComplex> A;

    EXPECT_DEATH(
        csr_to_dense(row_ptr, col_ind, vals, n, A),
        ""
    );
}

TEST(CsrToDenseDeath, AssertsOnNnzMismatch) {
    const int n = 2;
    std::vector<int> row_ptr{0,1,1}; // nnz=1
    std::vector<int> col_ind;        // size 0 -> mismatch
    std::vector<cuDoubleComplex> vals{C(1)};
    std::vector<cuDoubleComplex> A;

    EXPECT_DEATH(
        csr_to_dense(row_ptr, col_ind, vals, n, A),
        ""
    );
}
#endif

// Optional: round-trip with your dense_to_csr if available
// Converts dense -> CSR -> dense and checks equality.
extern void dense_to_csr(std::span<const cuDoubleComplex>, int,
                         std::vector<int>&, std::vector<int>&, std::vector<cuDoubleComplex>&, double);

TEST(CsrToDense, RoundTripDenseCsrDense) {
    const int n = 3;
    std::vector<cuDoubleComplex> dense{
        C(1), C(0), C(2),
        C(0), C(0), C(0),
        C(3), C(0), C(4),
    };

    std::vector<int> row_ptr, col_ind;
    std::vector<cuDoubleComplex> vals;
    dense_to_csr(std::span<const cuDoubleComplex>(dense), n, row_ptr, col_ind, vals, /*tol=*/0.0);

    std::vector<cuDoubleComplex> dense2;
    csr_to_dense(row_ptr, col_ind, vals, n, dense2);

    ASSERT_EQ(dense2.size(), dense.size());
    for (size_t i=0;i<dense.size();++i) {
        EXPECT_TRUE(CEq(dense2[i], dense[i])) << "Mismatch at idx " << i;
    }
}
