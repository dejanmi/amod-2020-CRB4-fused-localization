cmake_minimum_required(VERSION 2.8.3)
project(fused_localization)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  duckietown_msgs
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
  Pose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
 CATKIN_DEPENDS message_runtime
)

