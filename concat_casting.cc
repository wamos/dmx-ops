// g++ -std=c++11 -lrt -I ./eigen-3.4.0/ reshape_casting.cc -o reshape_casting
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
 
typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> Matrix_int8;
using Eigen::Tensor;

inline uint64_t getNanoSecond(struct timespec tp){
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (1000000000) * (uint64_t)tp.tv_sec + tp.tv_nsec;
}
 
int main() {
	struct timespec ts1,ts2;
	uint32_t iterations = 2;//1024*1024;
	uint32_t batch_size = 1;

	for(int iter = 0; iter < iterations; iter++){	
		auto start = getNanoSecond(ts1);	
		Eigen::Tensor<float, 3> input(128, 250, batch_size);
		Eigen::Tensor<float, 3> const2(128, 250, batch_size);
		Eigen::Tensor<float, 3> matmul(250, 250, batch_size);
		input.setConstant(2);
		const2.setConstant(0.5);
		Eigen::array<Eigen::Index, 3> extents = {1, 0, 2};
		input = input*const2;
		Eigen::Tensor<float, 3> output = input.shuffle(extents); // transpose of tensor along dim 0 and 1
		//std::cout << "dims:" << output.dimension(0) << "," << output.dimension(1) << "," << output.dimension(2) << std::endl;
		//output.setConstant(5);

		for(int batch = 0; batch < batch_size; batch++){
			Eigen::Tensor<float, 2> mat_a = input.chip(batch,2);
			Eigen::Tensor<float, 2> mat_b = output.chip(batch,2);
			Eigen::Tensor<float, 2> mat_ab = matmul.chip(batch,2);
			// std::cout << "mat_a:\n" << mat_a << std::endl;
			// std::cout << "mat_b:\n" << mat_b << std::endl;
			Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
			mat_ab = mat_a.contract(mat_b, product_dims);
			//std::cout << mat_ab.NumDimensions << "\n";
		}
		Eigen::array<uint32_t, 3> three_dims{1, 250*250, batch_size};
		matmul = matmul.reshape(three_dims);
		Eigen::Tensor<int8_t, 3> matmul_int8 = matmul.cast<int8_t>();
		//std::cout << "dims:" << matmul_int8.dimension(0) << "," << matmul_int8.dimension(1) << "," << matmul_int8.dimension(2) << std::endl;
		auto end = getNanoSecond(ts2);
		std::cout << end - start << "ns \n";
	}
}