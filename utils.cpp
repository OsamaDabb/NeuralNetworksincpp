#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include<random>
#include <iomanip>

#include "utils.h"

std::vector<std::vector<double>> values;
utils::Matrix::Matrix(int l, int w) : length(l), width(w), values(l, std::vector<double>(w, 0)) {};

void utils::Matrix::randomize(double mean, double var) {

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(mean, var);

	for (int i = 0; i < length; i++) {

		for (int j = 0; j < width; j++) {
			values[i][j] = distribution(generator);
		}
	}
}

void utils::Matrix::print() {


	for (std::vector<double> vec : values) {
		std::cout << "[ ";
		for (double val : vec) {
			std::cout << val << " ";
		}
		std::cout << "]" << std::endl;
	}

}

void utils::Matrix::print_size() {
	std::cout << "(" << length << ", " << width << ")" << std::endl;
}

void utils::print(std::vector<double> vec) {

	std::cout << "{ ";
	for (double val : vec) {
		std::cout << val << " ";
	}
	std::cout << "}" << std::endl;

}


utils::Matrix utils::multiply(const utils::Matrix& m1, const utils::Matrix& m2) {

	utils::Matrix result(m1.length, m2.width);

	for (int i = 0; i < m1.length; i++) {

		for (int c = 0; c < m2.width; c++) {

			for (int j = 0; j < m1.width; j++) {

				result.values[i][c] += m1.values[i][j] * m2.values[j][c];
			}
		}
	}

	return result;
}

std::vector<double> utils::multiply(const utils::Matrix& m1, const std::vector<double>& x) {

	std::vector<double> result(m1.length);

	for (int i = 0; i < m1.length; i++) {

		result[i] = 0;

		for (int j = 0; j < m1.width; j++) {

			result[i] += m1.values[i][j] * x[j];
		}

	}

	return result;
}

utils::Matrix utils::multiply(const utils::Matrix& m1, const double& c) {

	utils::Matrix result(m1.length, m1.width);

	for (int i = 0; i < m1.length; i++) {

		for (int j = 0; j < m1.width; j++) {

			result.values[i][j] = m1.values[i][j] * c;
		}

	}

	return result;
}

utils::Matrix utils::add(const utils::Matrix& m1, const utils::Matrix& m2) {

	utils::Matrix result(m1.length, m1.width);

	for (int i = 0; i < m1.length; i++) {
		for (int j = 0; j < m1.width; j++) {
			result.values[i][j] = m1.values[i][j] + m2.values[i][j];
		}
	}

	return result;
}

utils::Matrix utils::sub(const utils::Matrix& m1, const utils::Matrix& m2) {

	utils::Matrix result(m1.length, m1.width);

	for (int i = 0; i < m1.length; i++) {
		for (int j = 0; j < m1.width; j++) {
			result.values[i][j] = m1.values[i][j] - m2.values[i][j];
		}
	}

	return result;
}


std::vector<double> utils::add(const std::vector<double>& v1, const std::vector<double>& v2) {

	std::vector<double> result(v1.size());

	for (int i = 0; i < v1.size(); i++) {

		result[i] = v1[i] + v2[i];
	}

	return result;
}

std::vector<double> utils::sub(const std::vector<double>& v1, const std::vector<double>& v2) {

	std::vector<double> result(v1.size());

	for (int i = 0; i < v1.size(); i++) {

		result[i] = v1[i] - v2[i];
	}

	return result;
}

utils::Matrix utils::transpose(const utils::Matrix& input) {
	utils::Matrix result(input.width, input.length);

	for (int i = 0; i < input.length; i++) {
		for (int j = 0; j < input.width; j++) {
			result.values[j][i] = input.values[i][j];
		}
	}

	return result;
}

utils::Matrix utils::transpose(const std::vector<double>& input) {

	utils::Matrix result(1, input.size());

	for (int i = 0; i < input.size(); i++) {
		result.values[0][i] = input[i];
	}

	return result;
}

utils::Matrix utils::vecToMat(const std::vector<double>& input) {

	utils::Matrix result(input.size(), 1);

	for (int i = 0; i < input.size(); i++) {
		result.values[i][0] = input[i];
	}

	return result;

}

std::vector<double> utils::matToVec(const utils::Matrix& input) {

	std::vector<double> result(input.length, 0);

	for (int i = 0; i < input.length; i++) {
		result[i] = input.values[i][0];
	}

	return result;
}