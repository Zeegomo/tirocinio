#ifndef ANNT_NETWORK_RNN_H
#define ANNT_NETWORK_RNN_H

#include <ANNT.hpp>
#include "network.hpp"

class ANNT_Net : public Network {
	public:
		ANNT_Net(Config conf, std::vector<std::vector<double>> raw_time_series);
		Error train(bool verbose = true);
		std::pair<std::vector<double>, Error> evaluate();

	private:
		std::shared_ptr<ANNT::Neuro::XNeuralNetwork> net;
		std::shared_ptr<ANNT::Neuro::Training::XNetworkTraining> net_training;
};

#endif
