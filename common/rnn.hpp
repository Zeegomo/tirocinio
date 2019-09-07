#ifndef BASE_RNN_H
#define BASE_RNN_H

#include <vector>
#include <string>
#include "config.hpp"
#include "network.hpp"

struct Executor{
	Executor();
	std::pair<Config, std::vector<std::vector<double>>> get_data();
	void train(Network *model);
	void train(Network *model, std::string path);
	void evaluate(Network *model);
	void evaluate(Network *model, std::string path);

	private:
	Config conf;
	std::vector<std::vector<double>> raw_time_series;
	Error hist;
};

#endif
