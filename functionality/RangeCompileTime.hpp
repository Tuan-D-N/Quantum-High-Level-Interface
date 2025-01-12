#pragma once
#include <array>

template <int N>
struct rangeArray
{
    constexpr rangeArray(int start = 0) : arr()
    {
        for (auto i = start; i < N; ++i)
            arr[i] = i;
    }
    std::array<int, N> arr;
};

#define range(start, end) rangeArray<end>(start).arr