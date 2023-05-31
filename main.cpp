#include <cstdio>
#include <Kokkos_Core.hpp>

using view = Kokkos::View<double*>;

void init(view& array, int size) {
    Kokkos::parallel_for("init_view", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) = idx * 1.0/100.0;
    });
}

void add(view& array, double value, int size) {
    Kokkos::parallel_for("add_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) += value;
    });
}

void mul(view& array, double value, int size) {
    Kokkos::parallel_for("mul_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) *= value;
    });
}

void div(view& array, double value, int size) {
    Kokkos::parallel_for("div_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) /= value;
    });
}

void sqrt(view& array, int size) {
    Kokkos::parallel_for("sqrt_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) = Kokkos::sqrt(array(idx));
    });
}

void sin(view& array, int size) {
    Kokkos::parallel_for("sin_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) = Kokkos::sin(array(idx));
    });
}

void exp(view& array, int size) {
    Kokkos::parallel_for("exp_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) = Kokkos::exp(-array(idx));
    });
}

int main () {

    Kokkos::initialize();
    {
        constexpr int N = 1000 * 1000;
        auto A = view("A", N);
        init(A, N);

        add(A, 3.51, N);
        mul(A, 0.25, N);
        div(A, 1.23456789, N);
        sqrt(A, N);
        sin(A, N);
        exp(A, N);
    }
    Kokkos::finalize();

}
