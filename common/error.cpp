#include "error.hpp"
#include <iostream>
#include <math.h>
#include <numeric>

using namespace std;

void Error::add_record(vector<double> out, vector<double> expected){
	if(out.size() != expected.size()){
		cerr << "Invalid argument: size mismatch" << endl;
		cerr << "ignoring..." << endl;
		return;
	}

	for(int i = 0; i < out.size(); i++){

		double vbias, vmad, vmse, vrmse, vmape;
		vbias = vmad = vmse = vrmse = vmape = 0;

		vmad += fabs(out[i] - expected[i]);
		vbias += (out[i] - expected[i]);
		vmse += (out[i] - expected[i]) * (out[i] - expected[i]);
		vmape += fabs(out[i] - expected[i]) * 100 / expected[i];

		v_bias.push_back(vbias);
		v_mad.push_back(vmad);
		v_mse.push_back(vmse);
		v_mape.push_back(vmape);
	}
}

void Error::calc(){
	mse = accumulate(v_mse.begin(), v_mse.end(), 0) / v_mse.size();
	bias = accumulate(v_bias.begin(), v_bias.end(), 0) / v_bias.size();
	mad = accumulate(v_mad.begin(), v_mad.end(), 0) / v_mad.size();
	mape = accumulate(v_mape.begin(), v_mape.end(), 0) / v_mape.size();
	rmse = sqrt(mse);
}


