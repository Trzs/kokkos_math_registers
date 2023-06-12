#include <cstdio>
#include <Kokkos_Core.hpp>

using REAL = float;
using view = Kokkos::View<REAL*>;

void init(view& array, int size) {
    Kokkos::parallel_for("init_view", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) = idx * 1.0/100.0;
    });
}

void add(view& array, REAL value, int size) {
    Kokkos::parallel_for("add_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) += value;
    });
}

void add2(view& array, view& array2, int size) {
    Kokkos::parallel_for("add2_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) += array2(idx);
    });
}

void add3(view& array, view& array2, view& array3, int size) {
    Kokkos::parallel_for("add3_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) += array2(idx);
        array(idx) += array3(idx);
    });
}

void mul(view& array, REAL value, int size) {
    Kokkos::parallel_for("mul_kernel", size, KOKKOS_LAMBDA (const int idx) {
        array(idx) *= value;
    });
}

void div(view& array, REAL value, int size) {
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

void loop(view& array, int size, int iter) {
    Kokkos::parallel_for("loop_kernel", size, KOKKOS_LAMBDA (const int idx) {
        for (int i=0; i < iter; ++i) {
            array(idx) += idx + i;
        }
    });
}

void nested_loop( view& array, int size, int iterA, int iterB, int iterC, int iterD) {
    Kokkos::parallel_for("nested_loop_kernel", size, KOKKOS_LAMBDA (const int idx) {
        for (int i0=0; i0 < iterA; ++i0) {
            for (int i1=0; i1 < iterB; ++i1) {
                for (int i2=0; i2 < iterC; ++i2) {
                    for (int i3=0; i3 < iterD; ++i3) {
                        array(idx) += idx + i0*i3 - i1*i2;
                    }
                }
            }
        }
    });
}

int main () {

    Kokkos::initialize();
    {
        constexpr int N = 1000 * 1000;
        auto A = view("A", N);
        auto B = view("B", N);
        auto C = view("C", N);
        init(A, N);
        init(B, N);
        init(C, N);

        add(A, 3.51, N);
        add2(A, B, N);
        add3(A, B, C, N);
        mul(A, 0.25, N);
        div(A, 1.23456789, N);
        sqrt(A, N);
        sin(A, N);
        exp(A, N);
        loop(C, N, 25);
        nested_loop(B, N, 3, 3, 10, 25);
    }
    Kokkos::finalize();

}
