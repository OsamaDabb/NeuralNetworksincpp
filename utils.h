#pragma once
#ifndef __UTILS_H_INCLUDED__
#define __UTILS_H_INCLUDED__

namespace utils {

	class Matrix
	{
	public:
		int length, width;
		std::vector<std::vector<double>> values;
		Matrix(int l, int w);

		void randomize(double mean, double var);

		void print();

		void print_size();
	};

	void print(std::vector<double> vec);

	Matrix multiply(const Matrix& m1, const Matrix& m2);

	std::vector<double> multiply(const Matrix& m1, const std::vector<double>& x);

	Matrix multiply(const Matrix& m1, const double& c);

	Matrix add(const Matrix& m1, const Matrix& m2);

	Matrix sub(const Matrix& m1, const Matrix& m2);


	std::vector<double> add(const std::vector<double>& v1, const std::vector<double>& v2);

	std::vector<double> sub(const std::vector<double>& v1, const std::vector<double>& v2);

	Matrix transpose(const Matrix& input);

	Matrix transpose(const std::vector<double>& input);

	Matrix vecToMat(const std::vector<double>& input);

	std::vector<double> matToVec(const Matrix& input);

}





#endif // !__UTILS_H_INCLUDED__
