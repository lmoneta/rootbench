## If ROOT has been built with -Dsofie-sycl=On
You might want to specify the following CMake variables:
- IntelSYCL_DIR
- PORTBLAS_DIR
- PORTBLAS_INCLUDE_DIR
- PORTBLAS_SRC_DIR

You must also set CMAKE_CXX_COMPILER to an IntelSYCL compatible compiler (icpx, clang++, dpcpp).