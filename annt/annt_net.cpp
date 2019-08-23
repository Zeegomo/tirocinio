#include <ANNT.hpp>
#include <iostream>
#include "network.hpp"
#include "error.hpp"
#include "annt_net.hpp"

using namespace std;
using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

ANNT_Net::ANNT_Net(Config conf, vector<vector<double>> raw_time_series) : 
	Network(conf, raw_time_series) {
	net = make_shared<XNeuralNetwork>();

	net->AddLayer(make_shared<XLSTMLayer>(conf.input_dim, conf.hidden_dim));
        net->AddLayer(make_shared<XTanhActivation>());
        net->AddLayer(make_shared<XFullyConnectedLayer>(conf.hidden_dim, conf.output_dim));

	net_training = make_shared<XNetworkTraining>(net,
                              make_shared<XNesterovMomentumOptimizer>(conf.learning_rate),
                              make_shared<XMSECost>());

        net_training->SetTrainingSequenceLength(conf.batch_size);
}

Error ANNT_Net::train(bool verbose){
        auto samples = create_training_samples();	
	vector<fvector_t> inputs, outputs;

	for(auto vv : samples){
		for(int i = 0; i < vv.first.size(); i++){
			fvector_t tmp;
			for(int j = 0; j < vv.first[0].size(); j++){
				tmp.push_back(vv.first[i][j]);
			}
			inputs.push_back(tmp);
			outputs.push_back({(float)vv.second[i][0]});
		}
	}

	vector<double> hist;
        for (size_t epoch = 0; epoch < conf.num_epochs; epoch++){
            hist.push_back(net_training->TrainBatch(inputs, outputs));
            net_training->ResetState();

            if ((epoch % 100) == 0  && verbose){
                printf("Epoch %ld | MSE: %f \n", epoch, static_cast<float>(hist.back()));
            }
        }
	return err;
}

pair<vector<double>, Error> ANNT_Net::evaluate(){
	fvector_t network_output;
        fvector_t input(conf.input_dim);
        fvector_t output(1);

        for (size_t i = 0; i < time_series.size()-1; i++){
		for(int j = 0; j < conf.input_dim; j++){
			input[j] = time_series[i][j];
		}
		
        	net_training->Compute(input, output);
        	network_output.push_back(output[0]);
        }
	/*
        // now predict some points, which were excluded from training
        fvector_t network_prediction;
        ANNT::float_t error, minError = ANNT::float_t( 0.0 ), maxError = ANNT::float_t( 0.0 ), avgError = ANNT::float_t( 0.0 );

        // don't reset state of the recurrent network, just feed it the next point from the time series
        input[0] = timeSeries[timeSeries.size( ) - trainingParams.PredictionSize - 1];

        for ( size_t i = 0, j = timeSeries.size( ) - trainingParams.PredictionSize; i < trainingParams.PredictionSize; i++, j++ )
        {
            netTraining.Compute( input, output );
            networkPrediction[i] = output[0];

            // use just predicted point as the new input
            input[0] = output[0];

            // find prediction error
            error = fabs( output[0] - timeSeries[j] );
            avgError += error;

            if ( i == 0 )
            {
                minError = maxError = error;
            }
            else if ( error < minError )
            {
                minError = error;
            }
            else if ( error > maxError )
            {
                maxError = error;
            }
        }
	*
        avgError /= trainingParams.PredictionSize;
        printf( "Prediction error: min = %0.4f, max = %0.4f, avg = %0.4f \n",
                static_cast< float >( minError ), static_cast< float >( maxError ), static_cast< float >( avgError ) );

        // save training/prediction results into CSV file
        SaveData( trainingParams.OutputDataFile, timeSeries, networkOutput, networkPrediction );
	*/
	vector<double> res, expected;	
	for(int i = 0; i < time_series.size()-1; i++){
		res.push_back(network_output[i]);
		expected.push_back(time_series[i+1][conf.target_column]);
	}
	vector<double> rescaled = rescale(res);
	err.add_record(rescaled, expected);
	err.calc();
	return {rescaled, err};
}

