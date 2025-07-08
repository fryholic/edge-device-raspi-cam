#ifdef _OPENMP
#include <omp.h>
#include <iostream>
int main() {
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
    return 0;
}
#else
#include <iostream>
int main() {
    std::cout << "OpenMP not available" << std::endl;
    return 0;
}
#endif
