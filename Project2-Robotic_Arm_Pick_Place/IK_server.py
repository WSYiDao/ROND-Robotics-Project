#!/usr/bin/env python

# Copyright (C) 2017 Udacity Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from mpmath import *
from sympy import *


def getHomogeneousTransforms(alpha,a,d,q):
    T = Matrix([[             cos(q),            -sin(q),           0,              a],
                [  sin(q)*cos(alpha),  cos(q)*cos(alpha), -sin(alpha),  -sin(alpha)*d],
                [  sin(q)*sin(alpha),  cos(q)*sin(alpha),  cos(alpha),   cos(alpha)*d],
                [                  0,                  0,           0,               1]])
    return T

def rot_x(q):
    R_x = Matrix([[1, 0, 0],
                  [0, cos(q), -sin(q)],
                  [0, sin(q), cos(q)]])

    return R_x

def rot_y(q):
    R_y = Matrix([[cos(q), 0, sin(q)],
                  [0, 1, 0],
                  [-sin(q), 0, cos(q)]])

    return R_y

def rot_z(q):
    R_z = Matrix([[cos(q), -sin(q), 0],
                  [sin(q), cos(q), 0],
                  [0, 0, 1]])

    return R_z


def handle_calculate_IK(req):
    rospy.loginfo("Received %s eef-poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1
    else:

        ### Your FK code here
        # Create symbols
        # Create Modified DH parameters
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
        a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
        d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
        alpha,a,d,q = symbols("alpha a d q")
        q1, q2, q3, q4, q5, q6,q7 = symbols('q1:8')

        # Define Modified DH Transformation matrix
        s  = {alpha0: 0.,      a0: 0.,                   d1: 0.33+0.42,  q1: q1
            ,alpha1: -pi/2,   a1: 0.35,                 d2: 0.,         q2: q2 - pi/2.
            ,alpha2: 0.,      a2: 1.25,                 d3: 0.,         q3: q3
            ,alpha3: -pi/2,   a3: -0.054,               d4: 0.96+0.54,  q4: q4
            ,alpha4: pi/2,    a4: 0.,                   d5: 0.,         q5: q5
            ,alpha5: -pi/2,   a5: 0.,                   d6: 0.,         q6: q6
            ,alpha6: 0.,      a6: 0.,                   d7: 0.193+0.11, q7: 0.          }
        # Create individual transformation matrices
        # Extract rotation matrices from the transformation matrices
        T0_1 = getHomogeneousTransforms(alpha0,a0,d1,q1).subs(s)
        T1_2 = getHomogeneousTransforms(alpha1,a1,d2,q2).subs(s)
        T2_3 = getHomogeneousTransforms(alpha2,a2,d3,q3).subs(s)
        T3_4 = getHomogeneousTransforms(alpha3,a3,d4,q4).subs(s)
        T4_5 = getHomogeneousTransforms(alpha4,a4,d5,q5).subs(s)
        T5_6 = getHomogeneousTransforms(alpha5,a5,d6,q6).subs(s)
        T6_7 = getHomogeneousTransforms(alpha6,a6,d7,q7).subs(s)
        T0_3 = simplify(T0_1 * T1_2 * T2_3)
        T0_7 = simplify(T0_3 * T3_4 * T4_5 * T5_6 * T6_7)

        R01 = rot_x(alpha0) * rot_z(q1)
        R12 = rot_x(alpha1) * rot_z(q2)
        R23 = rot_x(alpha2) * rot_z(q3)
        R03 = simplify(R01 * R12 * R23)
        R03 = R03.subs(s)

        # Initialize service response
        joint_trajectory_list = []
        dist_error = 0
        for x in xrange(0, len(req.poses)):
            # IK code starts here
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation from request
            # px,py,pz = end-effector position
            # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            ### Your IK code here
            # Compensate for rotation discrepancy between DH parameters and Gazebo
            R_corr = rot_z(pi) * rot_y(-pi / 2)
            Rotation = rot_z(yaw) * rot_y(pitch) * rot_x(roll) * R_corr
            # Calculate joint angles using Geometric IK method
            wcx = px - 0.303 * Rotation[0, 2]
            wcy = py - 0.303 * Rotation[1, 2]
            wcz = pz - 0.303 * Rotation[2, 2]

            l23 = a2
            l35 = sqrt(a3**2 + d4**2)
            D1 = sqrt(wcx * wcx + wcy * wcy) - a1
            D2 = wcz - d1
            ww = D1**2 + D2**2
            w = sqrt(ww)

            theta1 = atan2(wcy,wcx)

            theta2 = -(atan2(D2, D1) + acos((l23**2 + ww- l35**2)/(2*l23*w)))
            theta2 = theta2.evalf(subs=s)
            theta2 = theta2 + pi/2

            theta3 = pi/2 - acos((l23**2 - ww + l35**2)/(2*l23*l35)) + atan2(a3, d4)
            theta3 = theta3.evalf(subs=s)

            R36 = simplify(R03.T * Rotation)
            R36 = R36.evalf(subs={q1: theta1, q2: theta2 , q3: theta3})

            if R36[1,2] == 1:
                theta4 = 0
                theta5 = 0
                theta6 = atan2(-R36[0,1],R36[0,0])
            elif R36[1,2] == -1:
                theta4 = 0
                theta5 = pi
                theta6 = atan2(R36[0,1], -R36[0,0])
            else:
                theta4 = atan2( R36[2,2], -R36[0,2])
                theta6 = atan2(-R36[1,1],  R36[1,0])
                theta5 = atan2(R36[1,0]/cos(theta6), R36[1,2])

            # Populate response for the IK request
            # In the next line replace theta1,theta2...,theta6 by your joint angle variables
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "Ready to receive an IK request"
    rospy.spin()

if __name__ == "__main__":
    IK_server()

