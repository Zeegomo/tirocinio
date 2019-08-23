#ifndef CONFIG_H
#define CONFIG_H

#include <set>

struct Config{
	int hidden_dim;
	int input_dim;
	int output_dim;

	int num_epochs;
	int batch_size;
	int n_sequences;
	float learning_rate;

	int target_column;
	std::set<int> ignore;
};

#endif
