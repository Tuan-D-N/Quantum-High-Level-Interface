#include <cmath>
#include <tuple>
#include <vector>
#include "GetAdjMat.hpp"
#include "Utilities.hpp"
#include "Linspace.hpp"
#include "OddRound.hpp"
#include <cassert>

using complex = std::complex<double>;

template <typename T, typename U>
std::tuple<T, U> createTuple(T slowChange, U fastChange) {
    return std::make_tuple(slowChange, fastChange);
}

double getMatAHelperDiffCorrel(double a, double b)
{
    return 1 - std::abs((a - b) / 2.0);
}

template<typename T, typename U>
std::vector<std::tuple<T, U>> allPermuteOfVectors(std::vector<T> slowChange, std::vector<U> fastChange, bool slowChangeFirst) {
    int length = slowChange.size() * fastChange.size();
    std::vector<std::tuple<double, double>> meshState(length);

    if (slowChangeFirst) {
        for (int i = 0; i < slowChange.size(); ++i)
        {
            for (int j = 0; j < fastChange.size(); ++j) //Can be parallelised
            {
                int index = i * fastChange.size() + j;
                meshState[index] = std::make_tuple(slowChange[i], fastChange[j]);
            }
        }
    }
    else {
        for (int i = 0; i < slowChange.size(); ++i)
        {
            for (int j = 0; j < fastChange.size(); ++j) //Can be parallelised
            {
                int index = i * fastChange.size() + j;
                meshState[index] = std::make_tuple(fastChange[j], slowChange[i]);
            }
        }
    }
    return meshState;
}

template std::vector<std::tuple<double, double>> allPermuteOfVectors<double, double>(std::vector<double>, std::vector<double>, bool);



std::vector<std::vector<complex>> getMatA(int evenQubits) {


    assert(isEven(evenQubits));

    int totalQubits = evenQubits + 1; //We need 1 extra qubit to get both sides
    int halfMatLength = pow(2, evenQubits); //At half length
    int fullMatLength = pow(2, totalQubits); //At full length

    int rLen = pow(2, evenQubits / 2); //Length of each list
    int thetaLen = pow(2, evenQubits / 2); //Happens to be square grid so they are the same
    int xLen = pow(2, evenQubits / 2);
    int yLen = pow(2, evenQubits / 2);

    int maxR = rLen - 1; // RList = [-3 -1 1 3] so length is 4

    auto xList = linspaceVec(-maxR, maxR, xLen); //Max X,Y is max R
    auto yList = linspaceVec(-maxR, maxR, yLen); //Max X,Y is max R
    auto thetaList = linspaceVec(0.0, M_PI, thetaLen); //Goes to Pi
    auto rList = linspaceVec(-maxR, maxR, rLen);


    //Makes list of rThetaStates: (Th1,R1), (Th1,R2) ... (Th2, R1), (Th2,R1) ... (Thn,Rn)
    auto rThetaStates = allPermuteOfVectors<double, double>(thetaList, rList, false);
    //Makes list of XYStates: (x1,y1), (x1,y2) ... (x2,y1),(x2,y2) ...
    auto xyStates = allPermuteOfVectors<double, double>(xList, yList, true);

    std::vector<std::tuple<double, double>>& rThetaInXY = rThetaStates;
    for (int i = 0; i < rThetaStates.size(); ++i) //This loop can be made parallel 
    {
        auto rThetaValue = rThetaStates[i];
        double r = std::get<0>(rThetaValue);
        double theta = std::get<1>(rThetaValue);
        rThetaInXY[i] = std::make_tuple(r * cos(theta), r * sin(theta));
    }

    //Converts XY coords to position on the the list
    auto toXYIndex = [xLen, maxR](int x, int y) {
        x = (x + maxR) / 2;
        y = (y + maxR) / 2;
        return xLen * x + y;
        };



    // Create a 2D vector of size x by y, initialized with 0
    std::vector<std::vector<complex>> aMat(fullMatLength, std::vector<complex>(fullMatLength, 0));

    for (int i = 0; i < halfMatLength; i++)
    {
        auto coords = rThetaInXY[i];
        double x, y;
        std::tie(x, y) = coords;

        double lx = roundToLowerOdd(x);
        double ux = roundToHigherOdd(x);
        double ly = roundToLowerOdd(y);
        double uy = roundToHigherOdd(y);

        std::vector<std::tuple<double, double>> allTargets{
            std::make_tuple(lx,ly),
            std::make_tuple(lx,uy),
            std::make_tuple(ux,ly),
            std::make_tuple(ux,uy),
        };

        for (auto target : allTargets) {
            double x_r, y_r;
            std::tie(x_r, y_r) = target;

            int xyIndex = toXYIndex(x_r, y_r);

            double value = getMatAHelperDiffCorrel(x_r, x) * getMatAHelperDiffCorrel(y_r, y);
            aMat[i + halfMatLength][xyIndex] = value;
            aMat[xyIndex][i + halfMatLength] = value;
        }
    }




    return aMat;
}


std::vector<std::vector<complex>> getMatAMini(int evenQubits) {


    assert(isEven(evenQubits));

    int halfMatLength = pow(2, evenQubits);
    int rLen = pow(2, evenQubits / 2);
    int thetaLen = pow(2, evenQubits / 2);
    int xLen = pow(2, evenQubits / 2);
    int yLen = pow(2, evenQubits / 2);

    int maxR = rLen - 1;

    auto xList = linspaceVec(-maxR, maxR, xLen);
    auto yList = linspaceVec(-maxR, maxR, yLen);
    auto thetaList = linspaceVec(0.0, M_PI, rLen);
    auto rList = linspaceVec(-maxR, maxR, rLen);

    std::vector<std::tuple<double, double>> rThetaStates;
    rThetaStates.reserve(rLen * thetaLen);
    for (auto thetaValue : thetaList)
    {
        for (auto rValue : rList)
        {
            rThetaStates.emplace_back(rValue, thetaValue);
        }
    }

    std::vector<std::tuple<double, double>> xyStates;
    xyStates.reserve(xLen * yLen);
    for (auto xValue : xList)
    {
        for (auto yValue : yList)
        {
            xyStates.emplace_back(xValue, yValue);
        }
    }

    std::vector<std::tuple<double, double>> rThetaInXY;
    rThetaInXY.reserve(rLen * thetaLen);
    for (auto rThetaValue : rThetaStates)
    {
        double r = std::get<0>(rThetaValue);
        double theta = std::get<1>(rThetaValue);
        rThetaInXY.emplace_back(r * cos(theta), r * sin(theta));
    }

    auto toXYIndex = [xLen, maxR](int x, int y) {
        assert(isOdd(x));
        assert(isOdd(y));
        x = (x + maxR) / 2;
        y = (y + maxR) / 2;
        return xLen * x + y;
        };



    // Create a 2D vector of size x by y, initialized with 0
    std::vector<std::vector<complex>> aMatMini(halfMatLength, std::vector<complex>(halfMatLength, 0));

    for (int i = 0; i < halfMatLength; i++)
    {
        auto coords = rThetaInXY[i];
        double x, y;
        std::tie(x, y) = coords;

        double lx = roundToLowerOdd(x);
        double ux = roundToHigherOdd(x);
        double ly = roundToLowerOdd(y);
        double uy = roundToHigherOdd(y);

        std::vector<std::tuple<double, double>> allTargets{
            std::make_tuple(lx,ly),
            std::make_tuple(lx,uy),
            std::make_tuple(ux,ly),
            std::make_tuple(ux,uy),
        };

        for (auto target : allTargets) {
            double x_r, y_r;
            std::tie(x_r, y_r) = target;

            int xyIndex = toXYIndex(x_r, y_r);

            double value = getMatAHelperDiffCorrel(x_r, x) * getMatAHelperDiffCorrel(y_r, y);

            aMatMini[xyIndex][i] = value;

        }

    }




    return aMatMini;
}
