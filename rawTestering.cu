#include <iostream>
#include <type_traits>

// Define precision enum using a list
#define PRECISION_X_LIST X(bit_32), X(bit_64)
#define X(name) name
enum class precision
{
    PRECISION_X_LIST
};
#undef X

// Macro to select the type based on precision
#define PRECISION_TYPE_COMPLEX(selectPrecision) typename std::conditional_t<selectPrecision == precision::bit_64, double, float>

// Template function to select precision-based type
template <precision a>
void foo(const PRECISION_TYPE_COMPLEX(a) f)
{
    std::cout << "Value: " << f << std::endl;
}

void v(){
    double g = 1;
    foo<precision::bit_64>(g);  // Correctly calls with 'double'
    
    float h = 1.0f;
    foo<precision::bit_32>(h);  // Correctly calls with 'float'
}

int main() {
    v();
    return 0;
}
