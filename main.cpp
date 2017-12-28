/**
 * Thomas Jacobs <hatsmagee@gmail.com>
 */
#include <map>
#include <string>
#include <stdio.h>

// Bullet
#include "../Utils/b3Clock.h"
#include "SharedMemory/PhysicsClientC_API.h"
#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3Quaternion.h"

// Tensorflow
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

// Bullet globals
const int CONTROL_RATE = 500;
b3PhysicsClientHandle kPhysClient = 0;
const b3Scalar FIXED_TIMESTEP = 1.0/((b3Scalar)CONTROL_RATE);
b3SharedMemoryCommandHandle command;
b3SharedMemoryStatusHandle statusHandle;
int statusType, ret;
b3JointInfo jointInfo;
b3JointSensorState state;
int robot;
map<std::string, int> jointNameToId;

// PD controller
double get_torque(double desired_position, double position, double &prev_error) {
	double error = (desired_position - position);
	double d = error - prev_error;
	double torque = 20.5 * error + 0.0 * d;
	prev_error = error;
	return torque;
}





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

// Runs a graph.
vector<Tensor> run_graph(Session* session, int input) {

  // Set up outputs
  vector<Tensor> resized_tensors;

  const Tensor& resized_tensor = resized_tensors[0];

  // Actually run the image through the model.
  string input_layer = "main_input";
  string output_layer0 = "output_0";
  string output_layer1 = "output_1";
  vector<Tensor> outputs;
  /*Status status = session->Run({{input_layer, resized_tensor}}, {output_layer0, output_layer1}, {}, &outputs);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }*/

  return outputs;
}




int main(int argc, char* argv[]) {

	// MacOS requires it run on the main thread
#ifdef __APPLE__
	kPhysClient = b3CreateInProcessPhysicsServerAndConnectMainThread(argc, argv);
#else
	kPhysClient = b3CreateInProcessPhysicsServerAndConnect(argc, argv);
#endif
	if (!kPhysClient)
		return -1;

	// Create visualizer
	command = b3InitConfigureOpenGLVisualizer(kPhysClient);
	b3ConfigureOpenGLVisualizerSetVisualizationFlags(command, COV_ENABLE_GUI, 0);
	b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	b3ConfigureOpenGLVisualizerSetVisualizationFlags(command, COV_ENABLE_SHADOWS, 0);
	b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	b3SetTimeOut(kPhysClient, 10);

	// This syncBodies is only needed when connecting to an existing physics server that has already some bodies
	command = b3InitSyncBodyInfoCommand(kPhysClient);
	statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	statusType = b3GetStatusType(statusHandle);

	// Set fixed time step
	command = b3InitPhysicsParamCommand(kPhysClient);
	ret = b3PhysicsParamSetTimeStep(command, FIXED_TIMESTEP);
	statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	ret = b3PhysicsParamSetRealTimeSimulation(command, false);
	statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

	// Gravity on
	ret = b3PhysicsParamSetGravity(command, 0, 0, -9.82);
	statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	b3Assert(b3GetStatusType(statusHandle) == CMD_CLIENT_COMMAND_COMPLETED);

	// Load the world
	command = b3LoadUrdfCommandInit(kPhysClient, "plane.urdf");
	b3LoadUrdfCommandSetUseFixedBase(command, true);	
	statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	statusType = b3GetStatusType(statusHandle);
	b3Assert(statusType == CMD_URDF_LOADING_COMPLETED);

	// Load robot
	command = b3LoadUrdfCommandInit(kPhysClient, "./ngr.urdf");
	int flags = URDF_USE_INERTIA_FROM_FILE;
	b3LoadUrdfCommandSetFlags(command, flags);
	b3LoadUrdfCommandSetUseFixedBase(command, false);

	// Set position
	b3LoadUrdfCommandSetStartPosition(command, 0, 0, 0.7);
	b3LoadUrdfCommandSetUseMultiBody(command, true);
	statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	statusType = b3GetStatusType(statusHandle);
	b3Assert(statusType == CMD_URDF_LOADING_COMPLETED);
	if (statusType == CMD_URDF_LOADING_COMPLETED) {
		robot = b3GetStatusBodyIndex(statusHandle);
	}

	// Disable default linear/angular damping
	b3SharedMemoryCommandHandle command = b3InitChangeDynamicsInfo(kPhysClient);
	double linearDamping = 0;
	double angularDamping = 0;
	b3ChangeDynamicsInfoSetLinearDamping(command, robot, linearDamping);
	b3ChangeDynamicsInfoSetAngularDamping(command, robot, angularDamping);
	statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

	// Loop through all joints
	int numJoints = b3GetNumJoints(kPhysClient, robot);
	for (int i=0; i<numJoints; ++i) {
		b3GetJointInfo(kPhysClient, robot, i, &jointInfo);
		if (jointInfo.m_jointName[0]) {
			jointNameToId[string(jointInfo.m_jointName)] = i;
		} else {
			continue;
		}

		// Reset before torque control - see #1459
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_VELOCITY);
		b3JointControlSetDesiredVelocity(command, jointInfo.m_uIndex, 0);
		b3JointControlSetMaximumForce(command, jointInfo.m_uIndex, 50);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	}


	// Initialize a tensorflow session
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return 1;
	}

	// Load graph
	const string& graph_file_name = "graph.pb";
	tensorflow::GraphDef graph_def;
	Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
	if (!load_graph_status.ok()) {
		printf("Failed to load compute graph.\n");
	}

	// Add the graph to the session
	printf("Creating compute graph.\n");
	status = session->Create(graph_def);
	printf("Creating compute graph...\n");
	if (!status.ok()) {
		printf("Failed to create compute graph.\n");
		std::cout << status.ToString() << "\n";
		return 1;
	}

	// Run
	std::vector<Tensor> outputs = run_graph(session, 0);


	// Loop
	unsigned long dtus1 = (unsigned long) 1000000.0*FIXED_TIMESTEP;
	double simTimeS = 0;
	double q[12], v[12];
	double prev_error[12];
	double torque;
	while (b3CanSubmitCommand(kPhysClient)) {
		simTimeS += 0.000005*dtus1;

		// Get joint values
		command = b3RequestActualStateCommandInit(kPhysClient, robot);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["1"], &state);
		q[0] = state.m_jointPosition;
		v[0] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["3"], &state);
		q[1] = state.m_jointPosition;
		v[1] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["5"], &state);
		q[2] = state.m_jointPosition;
		v[2] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["7"], &state);
		q[3] = state.m_jointPosition;
		v[3] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["0"], &state);
		q[4] = state.m_jointPosition;
		v[4] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["2"], &state);
		q[5] = state.m_jointPosition;
		v[5] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["4"], &state);
		q[6] = state.m_jointPosition;
		v[6] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["6"], &state);
		q[7] = state.m_jointPosition;
		v[7] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["8"], &state);
		q[8] = state.m_jointPosition;
		v[8] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["9"], &state);
		q[9] = state.m_jointPosition;
		v[9] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["10"], &state);
		q[10] = state.m_jointPosition;
		v[10] = state.m_jointVelocity;
		b3GetJointState(kPhysClient, statusHandle, jointNameToId["11"], &state);
		q[11] = state.m_jointPosition;
		v[11] = state.m_jointVelocity;

		// Abduction

		torque = get_torque(0 + 0.0 * sin(4*simTimeS), q[8], prev_error[8]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["8"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		torque = get_torque(0 + 0.0 * sin(4*simTimeS), q[9], prev_error[9]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["9"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		torque = get_torque(0 + 0.0 * sin(4*simTimeS), q[10], prev_error[10]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["10"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		torque = get_torque(0 + 0.0 * sin(4*simTimeS), q[11], prev_error[11]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["11"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		double stepCycle = fmod(simTimeS, 20.0);

		double stepProgress1 = 0;
//		if(stepCycle > 0 && stepCycle < 5)
		stepProgress1 = fmod(simTimeS/3*3.1415, 3.1415);
		double stepProgress2 = 0;
		if(stepCycle > 5 && stepCycle < 10)
			stepProgress2 = fmod(simTimeS/5*3.1415, 3.1415);
		double stepProgress3 = 0;
		if(stepCycle > 10 && stepCycle < 15)
			stepProgress3 = fmod(simTimeS/5*3.1415, 3.1415);
		double stepProgress4 = 0;
		if(stepCycle > 15 && stepCycle < 20)
			stepProgress4 = fmod(simTimeS/5*3.1415, 3.1415);

		// Knees

		// Apply some torque
		torque = get_torque(1.5 + 0.5 * sin(1*stepProgress1), q[0], prev_error[0]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["1"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(1.5 + 0.5 * sin(1*stepProgress2), q[1], prev_error[1]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["3"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(1.5 + 0.5 * sin(1*stepProgress3), q[2], prev_error[2]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["5"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(1.5 + 0.5 * sin(1*stepProgress4), q[3], prev_error[3]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["7"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);


		// Hips

		// Apply some torque
		torque = get_torque(0.6 + 0.3 * sin(1*stepProgress1), q[4], prev_error[4]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["0"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(0.6 + 0.3 * sin(1*stepProgress2), q[5], prev_error[5]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["2"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(0.4 + 0.3 * sin(1*stepProgress3), q[6], prev_error[6]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["4"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(0.4 + 0.3 * sin(1*stepProgress4), q[7], prev_error[7]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["6"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);


		// Run a step
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, b3InitStepSimulationCommand(kPhysClient));

		// debugging output
		//printf("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", simTimeS, q[0], q[1], v[0], v[1], torque);

		b3Clock::usleep(1000.*1000.*FIXED_TIMESTEP);
	}
	b3DisconnectSharedMemory(kPhysClient);
	return 0;
}

