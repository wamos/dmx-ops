// g++ -std=c++11 -lrt -I ./eigen-3.4.0/ mel_scale.cc -o mel_scale
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
 
typedef Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> Matrix_int8;

inline uint64_t getNanoSecond(struct timespec tp){
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return (1000000000) * (uint64_t)tp.tv_sec + tp.tv_nsec;
}


int main() {
	struct timespec ts1,ts2;
	uint32_t iterations = 1;//1024*1024;
	uint32_t batch_size = 1;	

	for(int iter = 0; iter < iterations; iter++){
		auto start = getNanoSecond(ts1);
		Eigen::Tensor<float, 3> FFTOuput(1025, 750, batch_size);
		Eigen::Tensor<float, 3> matmul(750, 1025, batch_size);
		Eigen::Tensor<float, 3> constmat(750, 1025, batch_size);	
		FFTOuput.setRandom();

		Eigen::array<Eigen::Index, 3> extents = {1, 0, 2};
		//auto FFTOuput_transposed = FFTOuput.transpose();
		matmul = FFTOuput.shuffle(extents); 
		std::cout << matmul.dimension(0) << "," << matmul.dimension(1) 
			<< "," << matmul.dimension(2) << "\n";
		
		matmul = matmul.pow(2);
		matmul = 700*matmul;		
		matmul = 1+matmul;
		matmul = matmul.unaryExpr([](float x){return std::log10(x);});
		matmul = 2595*matmul;
		Eigen::Tensor<int8_t, 3>  MFCCAudio_int8t = matmul.cast <int8_t> (); 	
		Eigen::array<uint32_t, 3> three_dims{750*1025, 1, batch_size};
		MFCCAudio_int8t = MFCCAudio_int8t.reshape(three_dims);
		// std::cout << MFCCAudio_int8t.NumDimensions << "\n";
		auto end = getNanoSecond(ts2);
		std::cout << end - start << "ns \n";
	}
}