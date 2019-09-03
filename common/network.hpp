#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>
#include "error.hpp"
#include "config.hpp"

class Network {
	public:
		virtual Error train(bool verbose = false) = 0;
		virtual std::pair<std::vector<double>, Error> evaluate() = 0;
		Network(Config config, std::vector<std::vector<double>> raw_time_series);
		virtual void save(std::string path) = 0;
		virtual void load(std::string path) = 0;
		//in place normalization of input time series
		void normalize_data();
		//in place diff of input time series
		void apply_diff();


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
		
		std::vector<std::vector<double>> rescale(std::vector<std::vector<double>> &source);
		std::vector<double> rescale(std::vector<double> source);
};

#endif
