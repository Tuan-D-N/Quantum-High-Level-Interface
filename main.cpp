#include <iostream>
#include "Runner/CircuitTest.hpp"
#include "Runner/ApplyIRadon.hpp"
#include "Runner/HHL_Helper.hpp"
#include "Runner/SparseGateSpeeds.hpp"
#include "CudaControl/Helper.hpp"
#include <cmath>

int main()
{
    main_runner();
    HHL_options option;

    option.num_phaseReg = 4;
    option.num_SYSReg = 3;
    option.num_ancilla = 1;
    option.t0 = 1;
    option.b = {cplx{9.02, 0},
                cplx{7.68, 0},
                cplx{8.16, 0},
                cplx{8.03, 0},
                cplx{1.61, 0},
                cplx{7.56, 0},
                cplx{6.38, 0},
                cplx{6.03, 0}};

    option.A = {cplx(6.9), cplx(1.04), cplx(3.66), cplx(5.07), cplx(2.84), cplx(9.52), cplx(8.56), cplx(6.53),
                cplx(1.04), cplx(5.88), cplx(1.06), cplx(4.1), cplx(0.36), cplx(6.79), cplx(0.27), cplx(1.43),
                cplx(3.66), cplx(1.06), cplx(6.81), cplx(7.26), cplx(9.18), cplx(4.23), cplx(7.48), cplx(0.7),
                cplx(5.07), cplx(4.1), cplx(7.26), cplx(6.14), cplx(0.84), cplx(2.64), cplx(3.5), cplx(4.03),
                cplx(2.84), cplx(0.36), cplx(9.18), cplx(0.84), cplx(2.62), cplx(3.5), cplx(9.64), cplx(2.68),
                cplx(9.52), cplx(6.79), cplx(4.23), cplx(2.64), cplx(3.5), cplx(3.04), cplx(5.81), cplx(9.94),
                cplx(8.56), cplx(0.27), cplx(7.48), cplx(3.5), cplx(9.64), cplx(5.81), cplx(1.27), cplx(7.23),
                cplx(6.53), cplx(1.43), cplx(0.7), cplx(4.03), cplx(2.68), cplx(9.94), cplx(7.23), cplx(5.41)};
    HHL_run(option);
    return 1;
}
