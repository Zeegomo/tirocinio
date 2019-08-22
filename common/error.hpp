#ifndef RNN_ERROR_H
#define RNN_ERROR_H

#include <vector>

struct Error {
	
	void add_record(std::vector<double> out, std::vector<double> expected);
	void calc();

	double mse, mad, bias, mape, rmse;

	std::vector<double> v_mse;
	std::vector<double> v_mad;
	std::vector<double> v_bias;
	std::vector<double> v_mape;
	std::vector<double> b_rmse;


};

#endif
