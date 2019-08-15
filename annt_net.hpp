#ifndef ANNT_NETWORK_RNN_H
#define ANNT_NETWORK_RNN_H

#include <ANNT.hpp>
#include "network.hpp"

class ANNT_Net : public Network {
	public:
		ANNT_Net(int num_epochs, int seq_len, int batch_size, int eval_batch_size, int n_sequences, float learning_rate, std::vector<double> raw_time_series, int input_dim, int hidden_dim, int output_dim);
		std::vector<double> train(bool verbose = true);
		std::pair<std::vector<double>, double> evaluate();

	private:
		std::shared_ptr<ANNT::Neuro::XNeuralNetwork> net;
		std::shared_ptr<ANNT::Neuro::Training::INetworkOptimizer> optimiser;
		std::shared_ptr<ANNT::Neuro::Training::ICostFunction> cost;
		ANNT::Neuro::Training::XNetworkTraining net_training;
};

#endif
