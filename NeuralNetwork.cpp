#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include<random>
#include <iomanip>


class Matrix
{
public:
	int length, width;
	std::vector<std::vector<double>> values;
	Matrix(int l, int w) : length(l), width(w), values(l, std::vector<double>(w, 0)) {};

	void randomize(double mean, double var) {

		std::default_random_engine generator;
		std::normal_distribution<double> distribution(mean, var);

		for (int i = 0; i < length; i++) {

			for (int j = 0; j < width; j++) {
				values[i][j] = distribution(generator);
			}
		}
	}

	void print() {


		for (std::vector<double> vec : values) {
			std::cout << "[ ";
			for (double val : vec) {
				std::cout << val << " ";
			}
			std::cout << "]" << std::endl;
		}

	}

	void print_size() {
		std::cout << "(" << length << ", " << width << ")" << std::endl;
	}
};

void print(std::vector<double> vec) {

	std::cout << "{ ";
	for (double val : vec) {
		std::cout << val << " ";
	}
	std::cout << "}" << std::endl;

}


Matrix multiply(const Matrix& m1, const Matrix& m2) {

	Matrix result(m1.length, m2.width);

	for (int i = 0; i < m1.length; i++) {

		for (int c = 0; c < m2.width; c++) {

			for (int j = 0; j < m1.width; j++) {

				result.values[i][c] += m1.values[i][j] * m2.values[j][c];
			}
		}
	}

	return result;
}

std::vector<double> multiply(const Matrix& m1, const std::vector<double>& x) {

	std::vector<double> result(m1.length);

	for (int i = 0; i < m1.length; i++) {

		result[i] = 0;

		for (int j = 0; j < m1.width; j++) {

			result[i] += m1.values[i][j] * x[j];
		}

	}

	return result;
}

Matrix multiply(const Matrix& m1, const double& c) {

	Matrix result(m1.length, m1.width);

	for (int i = 0; i < m1.length; i++) {

		for (int j = 0; j < m1.width; j++) {

			result.values[i][j] = m1.values[i][j] * c;
		}

	}

	return result;
}

Matrix add(const Matrix& m1, const Matrix& m2) {

	Matrix result(m1.length, m1.width);

	for (int i = 0; i < m1.length; i++) {
		for (int j = 0; j < m1.width; j++) {
			result.values[i][j] = m1.values[i][j] + m2.values[i][j];
		}
	}

	return result;
}

Matrix sub(const Matrix& m1, const Matrix& m2) {

	Matrix result(m1.length, m1.width);

	for (int i = 0; i < m1.length; i++) {
		for (int j = 0; j < m1.width; j++) {
			result.values[i][j] = m1.values[i][j] - m2.values[i][j];
		}
	}

	return result;
}


std::vector<double> add(const std::vector<double>& v1, const std::vector<double>& v2) {

	std::vector<double> result(v1.size());

	for (int i = 0; i < v1.size(); i++) {

		result[i] = v1[i] + v2[i];
	}

	return result;
}

std::vector<double> sub(const std::vector<double>& v1, const std::vector<double>& v2) {

	std::vector<double> result(v1.size());

	for (int i = 0; i < v1.size(); i++) {

		result[i] = v1[i] - v2[i];
	}

	return result;
}

std::vector<double> operator+(const std::vector<double>& v1, const std::vector<double>& v2) {
	return add(v1, v2);
}

Matrix transpose(const Matrix& input) {
	Matrix result(input.width, input.length);

	for (int i = 0; i < input.length; i++) {
		for (int j = 0; j < input.width; j++) {
			result.values[j][i] = input.values[i][j];
		}
	}

	return result;
}

Matrix transpose(const std::vector<double>& input) {

	Matrix result(1, input.size());

	for (int i = 0; i < input.size(); i++) {
		result.values[0][i] = input[i];
	}

	return result;
}

Matrix vecToMat(const std::vector<double>& input) {

	Matrix result(input.size(), 1);

	for (int i = 0; i < input.size(); i++) {
		result.values[i][0] = input[i];
	}

	return result;

}

std::vector<double> matToVec(const Matrix& input) {

	std::vector<double> result(input.length, 0);

	for (int i = 0; i < input.length; i++) {
		result[i] = input.values[i][0];
	}

	return result;
}

void ReLUOutput(std::vector<double>& input) {

	for (int i = 0; i < input.size(); i++) {

		input[i] = std::max(input[i], double(0));

	}
}

Matrix ReLUGradient(std::vector<double> prev_in, const Matrix& dldf) {

	Matrix result(dldf.length, 1);

	for (int i = 0; i < dldf.length; i++) {
		if (prev_in[i] > 0) {
			result.values[i][0] = dldf.values[i][0];
		}
		else {
			result.values[i][0] = 0;
		}
	}

	return result;
}


void sigmoidOutput(std::vector<double>& input) {

	for (int i = 0; i < input.size(); i++) {

		input[i] = 1 / (1 + exp(-input[i]));
	}

}

Matrix sigmoidGradient(std::vector<double> prev_in, const Matrix& dldf) {

	Matrix result(dldf.length, 1);

	std::vector<double> sig = prev_in;
	sigmoidOutput(sig);

	for (int i = 0; i < dldf.length; i++) {
		result.values[i][0] = sig[i] * (1 - sig[i]) * dldf.values[i][0];
	}

	return result;
}


class linearLayer {
public:
	linearLayer() = default;

	linearLayer(int in_size, int layer_size, std::string act, double use_moment) :
		weights(layer_size, in_size), biases(layer_size),
		w_gradients(layer_size, in_size), b_gradients(layer_size, 1), prev_in(in_size, 1), w_momentum(layer_size, in_size), b_momentum(layer_size, 1) {
		count = 0;
		input_size = in_size;
		output_size = layer_size;
		weights.randomize(0, sqrt(2.0 / double(in_size)));
		activation = act;
		momentum = use_moment;
		if (momentum > 0) {
			
		}
	}

	std::vector<double> output(const std::vector<double>& input) {

		count++;

		for (int i = 0; i < input.size(); i++) {

			prev_in.values[i][0] = input[i];

		}

		std::vector<double> result = multiply(weights, input) + biases;

		pre_activation = result;

		if (activation == "ReLU") {
			ReLUOutput(result);
		}
		else if (activation == "Sigmoid") {
			sigmoidOutput(result);
		}

		return result;
	}

	Matrix gradient(Matrix dldf) {

		if (activation == "ReLU") {
			dldf = ReLUGradient(pre_activation, dldf);
		}
		else if (activation == "Sigmoid") {
			dldf = sigmoidGradient(pre_activation, dldf);
		}

		Matrix grad = multiply(dldf, transpose(prev_in));

		w_gradients = add(w_gradients, grad);
		b_gradients = add(b_gradients, dldf);

		return multiply(transpose(weights), dldf);
	}

	void update_weights(double lr) {

		if (momentum > 0) {
			w_momentum = add(multiply(w_gradients, (1 - momentum)), multiply(w_momentum, momentum));
			b_momentum = add(multiply(b_gradients, (1 - momentum)), multiply(b_momentum, momentum));

			weights = sub(weights, multiply(w_momentum, lr / count));
			biases = sub(biases, matToVec(multiply(b_momentum, lr / count)));
		}
		else{
			weights = sub(weights, multiply(w_gradients, lr / count));
			biases = sub(biases, matToVec(multiply(b_gradients, lr / count)));
		}

		w_gradients = multiply(w_gradients, 0);
		b_gradients = multiply(b_gradients, 0);

		count = 0;
	}

	Matrix weights;
	std::vector<double> biases;

	int count;
	Matrix w_gradients;
	Matrix b_gradients;
	Matrix prev_in;
	std::vector<double> pre_activation;
	std::string activation;
	int input_size, output_size;
	double momentum;
	Matrix w_momentum;
	Matrix b_momentum;
};

//network class

class Network {
public:
	std::vector<linearLayer> layers;
	double lr, loss;
	std::string loss_f;
	Network(std::vector<linearLayer> l, double l_rate, std::string loss_func) : layers(l), lr(l_rate), loss(0) {
		loss_f = loss_func;
	};

	std::vector<double> output(std::vector<double> x) {

		for (int i = 0; i < layers.size(); i++) {

			x = layers[i].output(x);
		}

		return x;
	}

	void compute_gradient(double y_hat, double y) {

		Matrix dldf(1, 1);

		if (loss_f == "regression") {
			dldf.values[0] = { 2 * (y_hat - y) };
		}
		else if (loss_f == "binary") {
			dldf.values[0] = { -y /(y_hat) + (1 - y) / (1 - y_hat) };
		}

		for (int i = layers.size() - 1; i >= 0; i--) {

			dldf = layers[i].gradient(dldf);
		}
	}

	void compute_loss(double y_hat, double y) {

		if (loss_f == "binary") {
			loss += -y * log(y_hat) - (1 - y) * log(1 - y_hat);
		}
		else if (loss_f == "regression") {
			loss += pow((y - y_hat), 2);
		}
	}

	void update_weights() {

		std::cout << "Loss: " <<  loss / layers[0].count << std::endl;

		for (linearLayer& l : layers) {
			l.update_weights(lr);
		}
		loss = 0;
	}
};

int main() {

	std::cout << std::setprecision(2);

	std::vector<linearLayer> layers;
	int hidden_size = 32;

	linearLayer l1(2, hidden_size, "ReLU", 0.9);
	linearLayer l2(hidden_size, hidden_size, "ReLU", 0.9);
	linearLayer l3(hidden_size, 1, "Sigmoid", 0.9);

	layers.push_back(l1);
	layers.push_back(l2);
	layers.push_back(l3);

	Matrix data(1000, 2);
	data.randomize(0, 2);

	std::vector<double> labels(data.length, 0);

	
	for (int i = 0; i < data.length; i++) {
		if (pow(data.values[i][0], 3) + data.values[i][1] > 1) {
			labels[i] = 1;
		}
	}

	Network network(layers, 0.05, "binary");

	double y_hat;

	for (int ep = 0; ep < 50; ep++) {
		std::cout << "**EPOCH " << ep << "**" << std::endl;
		for (int i = 0; i < data.length; i++) {
			y_hat = network.output(data.values[i])[0];

			network.compute_loss(y_hat, labels[i]);
			network.compute_gradient(y_hat, labels[i]);


			if (ep % 10 == 0 && i % 24 == 0) {
				std::cout << "Input: ";
				print(data.values[i]);
				std::cout << "Output: " << y_hat << std::endl;

				/*
				cout << "Gradient: " << endl;
				for (linearLayer l : network.layers) {
					l.w_gradients.print();
				}
				cout << "network weights: " << endl;
				for (linearLayer l : network.layers) {
					l.weights.print();
				}*/
			}
		}

		network.update_weights();
		
		std::cout << std::endl;
	}
	
	return 0;
}
