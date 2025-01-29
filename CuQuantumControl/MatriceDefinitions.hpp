#pragma once
#include <cmath>
#include <type_traits>

#define HMat(...)                                                                 \
    {                                                                             \
        {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, { -INV_SQRT2, 0.0 } \
    }

#define XMat(...)                                        \
    {                                                    \
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, { 0.0, 0.0 } \
    }

#define YMat(...)                                         \
    {                                                     \
        {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, { 0.0, 0.0 } \
    }

#define ZMat(...)                                         \
    {                                                     \
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, { -1.0, 0.0 } \
    }

// Helper to select cos/sin based on type
template <typename T>
struct MathFunctions
{
    static inline T cos(T x) { return std::cos(x); }
    static inline T sin(T x) { return std::sin(x); }
};

template <>
struct MathFunctions<float>
{
    static inline float cos(float x) { return cosf(x); }
    static inline float sin(float x) { return sinf(x); }
};

// Generic RKMat macro (uses correct math functions for type)
#define RKMat(k, T)                                                          \
    {                                                                        \
        {1.0, 0.0},                                                          \
            {0.0, 0.0},                                                      \
            {0.0, 0.0},                                                      \
        {                                                                    \
            MathFunctions<T>::cos(2 * M_PI / (1 << static_cast<int>(k))),    \
                MathFunctions<T>::sin(2 * M_PI / (1 << static_cast<int>(k))) \
        }                                                                    \
    }

// RXMat, RYMat, and RZMat would be similarly adapted:

#define RXMat(theta, T)                               \
    {                                                 \
        {MathFunctions<T>::cos(theta / 2), 0.0},      \
            {0.0, -MathFunctions<T>::sin(theta / 2)}, \
            {0.0, -MathFunctions<T>::sin(theta / 2)}, \
        {                                             \
            MathFunctions<T>::cos(theta / 2), 0.0     \
        }                                             \
    }

#define RYMat(theta, T)                               \
    {                                                 \
        {MathFunctions<T>::cos(theta / 2), 0.0},      \
            {-MathFunctions<T>::sin(theta / 2), 0.0}, \
            {MathFunctions<T>::sin(theta / 2), 0.0},  \
        {                                             \
            MathFunctions<T>::cos(theta / 2), 0.0     \
        }                                             \
    }

#define RZMat(theta, T)                                                         \
    {                                                                           \
        {MathFunctions<T>::cos(-theta / 2), MathFunctions<T>::sin(-theta / 2)}, \
            {0.0, 0.0},                                                         \
            {0.0, 0.0},                                                         \
        {                                                                       \
            MathFunctions<T>::cos(theta / 2), MathFunctions<T>::sin(theta / 2)  \
        }                                                                       \
    }



    