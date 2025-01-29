#pragma once
#include <type_traits>

// Helper macros for conditional expansion
#define IF_ELSE(condition) IF_ELSE_##condition
#define IF_ELSE_0(...)
#define IF_ELSE_1(...) __VA_ARGS__
#define IF_ELSE_2(...) __VA_ARGS__
#define IF_ELSE_3(...) __VA_ARGS__
#define IF_ELSE_4(...) __VA_ARGS__
#define IF_ELSE_5(...) __VA_ARGS__

#define EXPAND_ARGS_0 
#define EXPAND_ARGS_1 double p1
#define EXPAND_ARGS_2 EXPAND_ARGS_1, double p2
#define EXPAND_ARGS_3 EXPAND_ARGS_2, double p3
#define EXPAND_ARGS_4 EXPAND_ARGS_3, double p4
#define EXPAND_ARGS_5 EXPAND_ARGS_4, double p5
// Extend as needed...

#define SELECT_ARGS(N) EXPAND_ARGS_##N
#define SELECT_EXTRA_ARGS(N) IF_ELSE(N)(, SELECT_ARGS(N))

#define EXPAND_VARS_0 
#define EXPAND_VARS_1 p1
#define EXPAND_VARS_2 EXPAND_VARS_1, p2
#define EXPAND_VARS_3 EXPAND_VARS_2, p3
#define EXPAND_VARS_4 EXPAND_VARS_3, p4
#define EXPAND_VARS_5 EXPAND_VARS_4, p5
#define SELECT_VARS(N) EXPAND_VARS_##N
#define SELECT_EXTRA_VARS(N) IF_ELSE(N)(, SELECT_VARS(N))


// how to use:
//
// #define FUNC(FUNC_NAME, NUMBEROFINPUTS)  \
//     void FUNC_NAME(SELECT_ARGS(NUMBEROFINPUTS)) \
//     {  \
//     }
//
// FUNC(hi,3) => void hi(int p1, int p2, int p3) { }
