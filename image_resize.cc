// g++ -std=c++11 -lrt -I ./eigen-3.4.0 image_resize.cc -o image_resize
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

inline uint64_t getNanoSecond(struct timespec tp){
    clock_gettime(CLOCK_REALTIME, &tp);
    return (1000000000) * (uint64_t)tp.tv_sec + tp.tv_nsec;
}

int main() {
	struct timespec ts1,ts2;
	uint32_t iterations = 2;//1024*1024;
	uint32_t batch_size = 1;		

	for(int iter = 0; iter < iterations; iter++){
		auto start = getNanoSecond(ts1);
		Eigen::Tensor<float, 2> kernel(2, 2);
		Eigen::Tensor<float, 3> output(540, 360, batch_size);
		//Eigen::Tensor<float, 3> output(360, 540, batch_size);
		Eigen::Tensor<float, 3> input(1080, 720, batch_size);
		//Eigen::Tensor<float, 3> input_t(720, 1080, batch_size);	
		input.setConstant(4);

		//Eigen::array<Eigen::Index, 3> extents = {1, 0, 2};
		//input_t = input.shuffle(extents); // transpose of tensor along dim 0 and 1
		//std::cout << input << "\n";
		//std::cout << "input dimensions\n";
		//std::cout << "rows:" << input.dimension(0) << ",cols:" << input.dimension(1) << "\n";
		// tensor slicing to take a 2*2 patch for max pooling
		Eigen::array<Eigen::Index, 3> extents = {2, 2, 1};
		for(int ch = 0; ch < batch_size; ch++){
			for(int i = 0; i < 4; i=i+2) {
				for(int j = 0; j < 4; j=j+2) {
					Eigen::array<Eigen::Index, 3> offsets = {i, j, ch};				
					input.unaryExpr([](float x){return x + 1;});
					Eigen::Tensor<float, 3> slice = input.slice(offsets, extents);
					Eigen::Tensor<float, 1> val = slice.maximum(Eigen::array<int, 2>({0, 1}));
					output(i/2, j/2, ch) = val(0);
				}
			}
		}
		//std::cout << output << "\n";
		//std::cout << "output dimensions\n";
		//std::cout << "rows:" << output.dimension(0) << ",cols:" << output.dimension(1) << "\n";
		auto end = getNanoSecond(ts2);
		std::cout << end - start << "ns \n";
	}
}
