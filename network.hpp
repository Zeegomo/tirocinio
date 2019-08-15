#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

class Network {
	public:
		virtual std::vector<double> train(bool verbose = false) = 0;
		virtual std::pair<std::vector<double>, double> evaluate() = 0;
		Network(int num_epochs, int seq_len, int batch_size, int eval_batch_size, int n_sequences, float learning_rate, std::vector<double> raw_time_series, int input_dim, int hidden_dim, int output_dim);

	protected:
		int num_epochs;
		int seq_len;
		int batch_size;
		int eval_batch_size;
		int n_sequences;
		int input_dim;
		int output_dim;
		int hidden_dim;
		float learning_rate;
		std::vector<double> raw_time_series;
};

#endif
