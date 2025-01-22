#pragma once

#define HMat                                                                      \
    {                                                                             \
        {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, {INV_SQRT2, 0.0}, { -INV_SQRT2, 0.0 } \
    }

#define XMat                                             \
    {                                                    \
        {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, { 0.0, 0.0 } \
    }

#define YMat                                              \
    {                                                     \
        {0.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}, { 0.0, 0.0 } \
    }

#define ZMat                                              \
    {                                                     \
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, { -1.0, 0.0 } \
    }

#define RKMat(k)                                               \
    {                                                          \
        {1.0, 0.0},                                            \
            {0.0, 0.0},                                        \
            {0.0, 0.0},                                        \
        {                                                      \
            cos(2 * M_PI / (1 << k)), sin(2 * M_PI / (1 << k)) \
        }                                                      \
    }

#define RXMat(theta)                \
    {                               \
        {cos(theta / 2), 0.0},      \
            {0.0, -sin(theta / 2)}, \
            {0.0, -sin(theta / 2)}, \
        {                           \
            cos(theta / 2), 0.0     \
        }                           \
    }

#define RYMat(theta)                \
    {                               \
        {cos(theta / 2), 0.0},      \
            {-sin(theta / 2), 0.0}, \
            {sin(theta / 2), 0.0},  \
        {                           \
            cos(theta / 2), 0.0     \
        }                           \
    }

#define RZMat(theta)                        \
    {                                       \
        {cos(-theta / 2), sin(-theta / 2)}, \
            {0.0, 0.0},                     \
            {0.0, 0.0},                     \
        {                                   \
            cos(theta / 2), sin(theta / 2)  \
        }                                   \
    }
