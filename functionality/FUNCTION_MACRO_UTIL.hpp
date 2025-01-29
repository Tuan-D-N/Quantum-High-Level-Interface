#pragma once
#include <type_traits>

// Helper macros for conditional expansion
#define IF_ELSE(condition) IF_ELSE_##condition
#define IF_ELSE_0(...)
#define IF_ELSE_1(...) __VA_ARGS__

#define EXPAND_ARGS_0 
#define EXPAND_ARGS_1 int p1
#define EXPAND_ARGS_2 EXPAND_ARGS_1, int p2
#define EXPAND_ARGS_3 EXPAND_ARGS_2, int p3
#define EXPAND_ARGS_4 EXPAND_ARGS_3, int p4
#define EXPAND_ARGS_5 EXPAND_ARGS_4, int p5
// Extend as needed...

#define SELECT_ARGS(N) EXPAND_ARGS_##N
#define SELECT_EXTRA_ARGS(N) IF_ELSE(N)(, SELECT_ARGS(N))

// how to use:
//
// #define FUNC(FUNC_NAME, NUMBEROFINPUTS)  \
//     void FUNC_NAME(SELECT_ARGS(NUMBEROFINPUTS)) \
//     {  \
//     }
//
// FUNC(hi,3) => void hi(int p1, int p2, int p3) { }
