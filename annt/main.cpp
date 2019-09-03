#include <iostream>
#include "rnn.hpp"
#include "annt_net.hpp"

using namespace std; 

int main(int argc, char *argv[]){
	Executor ex;
	auto data = ex.get_data();
	ANNT_Net model(data.first, data.second);
	if(argc > 1){
                cerr << "[ERR] ANNT does not support saving training state" << endl;
		exit(1);
        } else {
		cerr << "[WARN] ANNT does not support saving training state" << endl;
                ex.train(&model);
                ex.evaluate(&model);
        }
}
