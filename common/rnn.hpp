#ifndef BASE_RNN_H
#define BASE_RNN_H

#include <vector>
#include "config.hpp"
#include "network.hpp"

struct Executor{
	Executor();
	std::pair<Config, std::vector<std::vector<double>>> get_data();
	void start(Network *model);

	private:
	Config conf;
	std::vector<std::vector<double>> raw_time_series;
};

#endif
