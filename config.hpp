#ifndef CONFIG_H
#define CONFIG_H

struct Config{
	int hidden_dim;
	int input_dim;
	int output_dim;

	int num_epochs;
	int seq_len;
	int batch_size;
	int n_sequences;
	float learning_rate;
}

#endif