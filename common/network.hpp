#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "error.hpp"
#include "config.hpp"

class Network {
	public:
		virtual Error train(bool verbose = false) = 0;
		virtual std::pair<std::vector<double>, double> evaluate() = 0;
		Network(Config config, std::vector<std::vector<double>> raw_time_series);

	protected:
		Config conf;
		Error err;
		double mi, ma;
		std::vector<std::vector<double>> raw_time_series;
		std::vector<std::vector<double>> time_series;

		std::vector<std::pair<
			std::vector<std::vector<double>>,
			std::vector<std::vector<double>>
			>> create_training_samples(bool normalize = true, bool diff = true);
		
		//in place normalization of input time series
		void normalize_data();
		//in place diff of input time series
		void apply_diff();

		std::vector<std::vector<double>> rescale(std::vector<std::vector<double>> &source);
		std::vector<double> rescale(std::vector<double> source);
};

#endif
