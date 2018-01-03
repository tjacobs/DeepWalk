# DeepWalk CMake file.

# We are modern
cmake_minimum_required (VERSION 3.5)

# Include Bullet
include_directories(
	bullet3
	bullet3/examples
	bullet3/examples/ThirdPartyLibs
	bullet3/src
)

# Add bullet
add_subdirectory(bullet3)

# Warn
set(CXX_FLAGS "-Wall ")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

# Needed to include TensorFlow includes
add_definitions(-std=c++11 -Wno-invalid-partial-specialization -Wc++11-narrowing)

# Include local TensorFlow
include_directories(/usr/local/include)
include_directories(/usr/local/include/google/tensorflow)
include_directories(src/Eigen-3.3)
include_directories(.)
link_directories(/usr/local/lib)

# Include OpenGL
if(APPLE)
	find_library(COCOA NAMES Cocoa)
	link_libraries(${COCOA} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
endif(APPLE)

# Sources
set(BULLET_PHYSICS_SOURCE_DIR "bullet3")
set(DeepWalk_SRCS
	src/main.cpp
	src/tensorflow.cpp
	src/gui/InProcessExampleBrowser.cpp

	# TinyRenderer
	${BULLET_PHYSICS_SOURCE_DIR}/examples/TinyRenderer/geometry.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/TinyRenderer/model.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/TinyRenderer/tgaimage.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/TinyRenderer/our_gl.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/TinyRenderer/TinyRenderer.cpp

	${BULLET_PHYSICS_SOURCE_DIR}/examples/RobotSimulator/b3RobotSimulatorClientAPI.cpp

	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/IKTrajectoryHelper.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/TinyRendererVisualShapeConverter.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/InProcessMemory.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsClient.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsServer.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsServerExample.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsServerExampleBullet2.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/SharedMemoryInProcessPhysicsC_API.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsServerSharedMemory.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsDirect.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsServerCommandProcessor.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/b3PluginManager.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsClientSharedMemory.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PhysicsClientC_API.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/Win32SharedMemory.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/SharedMemory/PosixSharedMemory.cpp

	${BULLET_PHYSICS_SOURCE_DIR}/examples/Utils/RobotLoggingUtil.cpp

	${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/tinyxml/tinystr.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/tinyxml/tinyxml.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/tinyxml/tinyxmlerror.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/tinyxml/tinyxmlparser.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/Wavefront/tiny_obj_loader.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/ThirdPartyLibs/stb_image/stb_image.cpp

	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportColladaDemo/LoadMeshFromCollada.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportObjDemo/LoadMeshFromObj.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportObjDemo/Wavefront2GLInstanceGraphicsShape.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportMJCFDemo/BulletMJCFImporter.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportURDFDemo/BulletUrdfImporter.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportURDFDemo/MyMultiBodyCreator.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportURDFDemo/URDF2Bullet.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportURDFDemo/UrdfParser.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportURDFDemo/urdfStringSplit.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/Importers/ImportMeshUtility/b3ImportMeshUtility.cpp

	${BULLET_PHYSICS_SOURCE_DIR}/examples/MultiThreading/b3PosixThreadSupport.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/MultiThreading/b3Win32ThreadSupport.cpp
	${BULLET_PHYSICS_SOURCE_DIR}/examples/MultiThreading/b3ThreadSupportInterface.cpp	
)

# Executable
add_executable(DeepWalk ${DeepWalk_SRCS})

# Link
target_link_libraries(DeepWalk BulletExampleBrowserLib BulletFileLoader BulletWorldImporter BulletSoftBody BulletDynamics BulletCollision BulletInverseDynamicsUtils BulletInverseDynamics LinearMath OpenGLWindow gwen BussIK Bullet3Common tensorflow_cc tensorflow_framework)

