/**
 * Thomas Jacobs <hatsmagee@gmail.com>
 */
#include <map>
#include <string>
#include <stdio.h>

// Bullet
#include "bullet3/examples/Utils/b3Clock.h"
#include "bullet3/examples/SharedMemory/PhysicsClientC_API.h"
#include "bullet3/src/Bullet3Common/b3Vector3.h"
#include "bullet3/src/Bullet3Common/b3Quaternion.h"
#include "bullet3/examples/RobotSimulator/b3RobotSimulatorClientAPI.h"

// Tensorflow
#include "tensorflow.h"

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
	double torque = 70.0 * error + 50.0 * d;
	prev_error = error;
	return torque;
}

// Main
int main(int argc, char* argv[]) {

	// Set up Bullet simulator
	b3RobotSimulatorClientAPI* sim = new b3RobotSimulatorClientAPI();
	sim->connect(eCONNECT_GUI);
	sim->configureDebugVisualizer(COV_ENABLE_GUI, 0);
	sim->setTimeOut(10);
	sim->syncBodies();
	b3Scalar fixedTimeStep = 1./240.;
	sim->setTimeStep(fixedTimeStep);
	sim->setGravity(b3MakeVector3(0, 0, -9.8));
	sim->loadURDF("../data/terrain.urdf");

	// Load robot
	kPhysClient = sim->getClientHandle();
	command = b3LoadUrdfCommandInit(kPhysClient, "../data/ngr.urdf");
	int flags = URDF_USE_INERTIA_FROM_FILE;
	b3LoadUrdfCommandSetFlags(command, flags);
	b3LoadUrdfCommandSetUseFixedBase(command, false);

	// Set position
	b3LoadUrdfCommandSetStartPosition(command, 0, 0, 0.4);
	b3LoadUrdfCommandSetUseMultiBody(command, true);
	statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
	statusType = b3GetStatusType(statusHandle);
	b3Assert(statusType == CMD_URDF_LOADING_COMPLETED);
	if (statusType == CMD_URDF_LOADING_COMPLETED) {
		robot = b3GetStatusBodyIndex(statusHandle);
	}

	// Disable default linear/angular damping
	b3SharedMemoryCommandHandle command = b3InitChangeDynamicsInfo(kPhysClient);
	double linearDamping = 10;
	double angularDamping = 10;
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
		b3JointControlSetMaximumForce(command, jointInfo.m_uIndex, 100);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);
		//printf("%d %s\n", i, jointInfo.m_jointName);

		command = b3InitChangeDynamicsInfo(kPhysClient);
		b3ChangeDynamicsInfoSetLateralFriction(command, robot, jointInfo.m_uIndex, 1.0);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

	}

	// Load TensorFlow graph
	Session* session = load_graph("../data/graph.pb");

	// Load input
	vector<float> inputs;
	for(int i = 0; i < 8; i++)
		inputs.push_back(0);
	inputs.push_back(fmod(0/3*3.1415, 3.1415));

	// Run graph
	vector<float> outputs_v = run_graph(session, inputs);

	// Output graph
	cout << "Output: ";
	for(int i = 0; i < 8; i++)
		cout << outputs_v[i] <<  "  ";
	cout << endl;

	// Load output
	vector<float> outputs;
	for(int i = 0; i < 8; i++)
		outputs.push_back(0.0);

	// Stand
	outputs[1] = 1.5;
	outputs[5] = 0.5;
	outputs[3] = 1.5;
	outputs[7] = 0.5;

	outputs[0] = 1.5;
	outputs[4] = 0.5;
	outputs[2] = 1.5;
	outputs[6] = 0.5;

	// Loop
	unsigned long dtus1 = (unsigned long) 1000000.0*FIXED_TIMESTEP;
	double simTimeS = 0;
	double q[12], v[12];
	double prev_error[12];
	double torque;
	int rotateCamera = 0;

	while (b3CanSubmitCommand(kPhysClient)) {
		simTimeS += 0.000001*dtus1;

		// Keyboard
		b3KeyboardEventsData keyEvents;
		sim->getKeyboardEvents(&keyEvents);
		if (keyEvents.m_numKeyboardEvents)
		{
			for (int i=0;i<keyEvents.m_numKeyboardEvents;i++)
			{
				b3KeyboardEvent& e = keyEvents.m_keyboardEvents[i];

				if (e.m_keyCode == 'r' && e.m_keyState&eButtonTriggered)
				{
					rotateCamera = 1-rotateCamera;
				}
//				printf("keyEvent[%d].m_keyCode = %d, state = %d\n", i,keyEvents.m_keyboardEvents[i].m_keyCode,keyEvents.m_keyboardEvents[i].m_keyState);
			}
		}

		// Rotating camera?
		if (rotateCamera)
		{
			static double yaw = 0;
			double distance = 1;
			yaw += 0.1;
			b3Vector3 basePos;
			b3Quaternion baseOrn;
			sim->getBasePositionAndOrientation(robot, basePos, baseOrn);
			sim->resetDebugVisualizerCamera(distance, -20, yaw, basePos);
		}


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

		double speed = 1.0;
		double stepCycle = fmod(simTimeS * speed, 4.0);

		float a = 0.0;
		float p = 200 / speed;

		if(stepCycle >= 0 && stepCycle < 1) {
			outputs[0] += (2.0 - outputs[0])/p;
			outputs[4] += (1.5 + a - outputs[4])/p;
		}
		if(stepCycle >= 1 && stepCycle < 2) {
			outputs[0] += (2.0 - outputs[0])/p;
			outputs[4] += (0.5 + a - outputs[4])/p;
		}
		if(stepCycle >= 2 && stepCycle < 3) {
			outputs[0] += (1.5 - outputs[0])/p;
			outputs[4] += (0.5 + a - outputs[4])/p;
		}
		if(stepCycle >= 3 && stepCycle < 4) {
			outputs[0] += (1.0 - outputs[0])/p;
			outputs[4] += (1.5 + a - outputs[4])/p;
		}

		if(stepCycle >= 0 && stepCycle < 1) {
			outputs[1] += (1.5 - outputs[1])/p;
			outputs[5] += (0.5 + a - outputs[5])/p;
		}
		if(stepCycle >= 1 && stepCycle < 2) {
			outputs[1] += (1.0 - outputs[1])/p;
			outputs[5] += (1.5 + a - outputs[5])/p;
		}
		if(stepCycle >= 2 && stepCycle < 3) {
			outputs[1] += (2.0 - outputs[1])/p;
			outputs[5] += (1.5 + a - outputs[5])/p;
		}
		if(stepCycle >= 3 && stepCycle < 4) {
			outputs[1] += (2.0 - outputs[1])/p;
			outputs[5] += (0.5 + a - outputs[5])/p;
		}

		if(stepCycle >= 0 && stepCycle < 1) {
			outputs[2] += (1.5 - outputs[2])/p;
			outputs[6] += (0.5 + a - outputs[6])/p;;
		}
		if(stepCycle >= 1 && stepCycle < 2) {
			outputs[2] += (1.0 - outputs[2])/p;
			outputs[6] += (1.5 + a - outputs[6])/p;
		}
		if(stepCycle >= 2 && stepCycle < 3) {
			outputs[2] += (2.0 - outputs[2])/p;
			outputs[6] += (1.5 + a - outputs[6])/p;
		}
		if(stepCycle >= 3 && stepCycle < 4) {
			outputs[2] += (2.0 - outputs[2])/p;
			outputs[6] += (0.5 + a - outputs[6])/p;
		}

		if(stepCycle >= 0 && stepCycle < 1) {
			outputs[3] += (2.0 - outputs[3])/p;
			outputs[7] += (1.5 + a - outputs[7])/p;
		}
		if(stepCycle >= 1 && stepCycle < 2) {
			outputs[3] += (2.0 - outputs[3])/p;
			outputs[7] += (0.5 + a - outputs[7])/p;
		}
		if(stepCycle >= 2 && stepCycle < 3) {
			outputs[3] += (1.5 - outputs[3])/p;
			outputs[7] += (0.5 + a - outputs[7])/p;
		}
		if(stepCycle >= 3 && stepCycle < 4) {
			outputs[3] += (1.0 - outputs[3])/p;
			outputs[7] += (1.5 + a - outputs[7])/p;
		}


		// Knees

		// Apply some torque
		torque = get_torque(outputs[0], q[0], prev_error[0]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["1"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(outputs[1], q[1], prev_error[1]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["3"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(outputs[2], q[2], prev_error[2]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["5"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(outputs[3], q[3], prev_error[3]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["7"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);


		// Hips

		// Apply some torque
		torque = get_torque(outputs[4], q[4], prev_error[4]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["0"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(outputs[5], q[5], prev_error[5]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["2"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(outputs[6], q[6], prev_error[6]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["4"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);

		// Apply some torque
		torque = get_torque(outputs[7], q[7], prev_error[7]);
		b3GetJointInfo(kPhysClient, robot, jointNameToId["6"], &jointInfo);
		command = b3JointControlCommandInit2(kPhysClient, robot, CONTROL_MODE_TORQUE);
		b3JointControlSetDesiredForceTorque(command, jointInfo.m_uIndex, torque);
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, command);


		// Run a step
		statusHandle = b3SubmitClientCommandAndWaitStatus(kPhysClient, b3InitStepSimulationCommand(kPhysClient));

		// Sleep
		b3Clock::usleep(1000.*1000.*FIXED_TIMESTEP);
	}
	b3DisconnectSharedMemory(kPhysClient);
	return 0;
}

