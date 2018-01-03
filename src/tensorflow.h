#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/ops/image_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/lib/io/path.h>
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::Session;
using tensorflow::string;
using tensorflow::int32;
using namespace tensorflow;
using namespace tensorflow::ops;
using namespace std;

Status LoadGraph(const string& graph_file_name, std::unique_ptr<tensorflow::Session>* session);
Session* load_graph(const string& graph_file_name);
vector<float> run_graph(Session* session, vector<float> inputs);
