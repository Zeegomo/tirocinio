#include <ANNT.hpp>
#include "network.hpp"
#include "annt_net.hpp"

using namespace std;
using namespace ANNT;
using namespace ANNT::Neuro;
using namespace ANNT::Neuro::Training;

ANNT_Net::ANNT_Net(int num_epochs, int seq_len, int batch_size, int eval_batch_size, int n_sequences, float learning_rate, std::vector<double> raw_time_series, int input_dim, int hidden_dim, int output_dim) : 
	Network(num_epochs, seq_len, batch_size, eval_batch_size, n_sequences, learning_rate, raw_time_series, input_dim, hidden_dim, output_dim), 
	net(new XNeuralNetwork()),
	optimiser(new XNesterovMomentumOptimizer(learning_rate)),
	cost(new XMSECost()),
      	net_training(net, optimiser, cost) {
        net->AddLayer(make_shared<XLSTMLayer>(input_dim, hidden_dim));
        net->AddLayer(make_shared<XTanhActivation>());
        net->AddLayer(make_shared<XFullyConnectedLayer>(hidden_dim, output_dim));
	
        net_training.SetTrainingSequenceLength(batch_size);
}

vector<double> ANNT_Net::train(bool verbose){
        vector<fvector_t> inputs, outputs;
	vector<double> hist;
        for (size_t i = 0; i < n_sequences; i++){
            for (size_t j = 0; j < batch_size; j++){
                inputs.push_back({raw_time_series[i + j]});
                outputs.push_back({raw_time_series[i + j + 1]});
            }
        }

        for (size_t epoch = 1; epoch <= num_epochs; epoch++){
            hist.push_back(net_training.TrainBatch(inputs, outputs));
            net_training.ResetState();

            if ((epoch % 100) == 0  && verbose){
                printf("%0.4f ", static_cast<float>(hist.back()));
            }
        }
	return hist;
}

pair<vector<double>, double> ANNT_Net::evaluate(){
	fvector_t network_output;
        fvector_t input(1);
        fvector_t output(1);

	double avg_error = 0;
        for (size_t i = 0; i < seq_len-1; i++){
            input[0] = raw_time_series[i]; // here input always comes from original time series

            net_training.Compute(input, output);
            network_output.push_back(output[0]);
	    avg_error += (raw_time_series[i]-output[0])*(raw_time_series[i]-output[0]);
        }
	avg_error /= raw_time_series.size();
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
	vector<double> res;
	for(int i = 0; i < seq_len-1; i++){
		res.push_back(network_output[i]);
	}
	return {res, avg_error};
}

