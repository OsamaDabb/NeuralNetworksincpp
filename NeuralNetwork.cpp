#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include<random>
#include <iomanip>

#include "utils.h"

void ReLUOutput(std::vector<double>& input) {

	for (int i = 0; i < input.size(); i++) {

		input[i] = std::max(input[i], double(0));

	}
}

utils::Matrix ReLUGradient(std::vector<double> prev_in, const utils::Matrix& dldf) {

	utils::Matrix result(dldf.length, 1);

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

utils::Matrix sigmoidGradient(std::vector<double> prev_in, const utils::Matrix& dldf) {

	utils::Matrix result(dldf.length, 1);

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

		std::vector<double> result = utils::add(multiply(weights, input), biases);

		pre_activation = result;

		if (activation == "ReLU") {
			ReLUOutput(result);
		}
		else if (activation == "Sigmoid") {
			sigmoidOutput(result);
		}

		return result;
	}

	utils::Matrix gradient(utils::Matrix dldf) {

		if (activation == "ReLU") {
			dldf = ReLUGradient(pre_activation, dldf);
		}
		else if (activation == "Sigmoid") {
			dldf = sigmoidGradient(pre_activation, dldf);
		}

		utils::Matrix grad = multiply(dldf, transpose(prev_in));

		w_gradients = add(w_gradients, grad);
		b_gradients = add(b_gradients, dldf);

		return multiply(transpose(weights), dldf);
	}

	void update_weights(double lr) {

		if (momentum > 0) {
			w_momentum = utils::add(utils::multiply(w_gradients, (1 - momentum)), utils::multiply(w_momentum, momentum));
			b_momentum = utils::add(utils::multiply(b_gradients, (1 - momentum)), utils::multiply(b_momentum, momentum));

			weights = utils::sub(weights, utils::multiply(w_momentum, lr / count));
			biases = utils::sub(biases, utils::matToVec(multiply(b_momentum, lr / count)));
		}
		else{
			weights = utils::sub(weights, utils::multiply(w_gradients, lr / count));
			biases = utils::sub(biases, utils::matToVec(multiply(b_gradients, lr / count)));
		}

		w_gradients = utils::multiply(w_gradients, 0);
		b_gradients = utils::multiply(b_gradients, 0);

		count = 0;
	}

	utils::Matrix weights;
	std::vector<double> biases;

	int count;
	utils::Matrix w_gradients;
	utils::Matrix b_gradients;
	utils::Matrix prev_in;
	std::vector<double> pre_activation;
	std::string activation;
	int input_size, output_size;
	double momentum;
	utils::Matrix w_momentum;
	utils::Matrix b_momentum;
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

		utils::Matrix dldf(1, 1);

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

	utils::Matrix data(1000, 2);
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
				utils::print(data.values[i]);
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
