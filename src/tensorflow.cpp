#include "tensorflow.h"
using namespace std;

// Reads a model graph definition from disk, and creates a session object you can use to run it.
Status LoadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '", graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}

Session* load_graph(const string& graph_file_name) {

	// Initialize a tensorflow session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		cout << status.ToString() << "\n";
		return 0;
	}

	// Load graph
	tensorflow::GraphDef graph_def;
	status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!status.ok()) {
		printf("Failed to load compute graph.\n");
		cout << status.ToString() << "\n";
		return 0;
	}

	// Add the graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) {
		printf("Failed to create compute graph.\n");
		cout << status.ToString() << "\n";
		return 0;
	}

	return session;
}

// Runs a graph.
vector<float> run_graph(Session* session, vector<float> inputs) {

	// Create tensor
	Tensor image_tensor(DT_FLOAT, TensorShape({1, 9}));
	auto input_tensor_mapped = image_tensor.tensor<float, 2>();
	for(int i = 0; i < 9; i++)
	 	input_tensor_mapped(i) = inputs[i];

	// Run
	string input_layer = "main_input";
	string output_layer = "output_0";
	vector<Tensor> outputs;
	Status status = session->Run({{input_layer, image_tensor}}, {output_layer}, {}, &outputs);
	if (!status.ok()) {
		LOG(ERROR) << status;
	}

	// Save
	vector<float> outputs_vector;
	for(int i = 0; i < 8; i++)
		outputs_vector.push_back(0.0 + outputs[0].flat<float>()(i));
	return outputs_vector;
}
