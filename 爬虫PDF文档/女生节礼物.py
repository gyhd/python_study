#!/usr/bin/env python
#coding=utf-8
#女生节礼物
 
import rospy
from sensor_msgs.msg import LaserScan
import numpy
import copy
 
node_name = "Test_Maker"
 
class Test_Maker():
    def __init__(self):
        self.Define()
        rospy.Timer(rospy.Duration(0.5), self.Timer_CB1)
        rospy.Timer(rospy.Duration(0.5), self.Timer_CB2)
        rospy.Timer(rospy.Duration(0.5), self.Timer_CB3)
        rospy.Timer(rospy.Duration(0.5), self.Timer_CB4)
        rospy.spin()
 
    def Define(self):
        self.pub_scan1 = rospy.Publisher('test/test_scan1', LaserScan, queue_size=1)
        self.pub_scan2 = rospy.Publisher('test/test_scan2', LaserScan, queue_size=1)
        self.pub_scan3 = rospy.Publisher('test/test_scan3', LaserScan, queue_size=1)
        #慎用！！！！
        self.pub_scan4 = rospy.Publisher('test/test_scan4', LaserScan, queue_size=1)
 
    def Timer_CB1(self, e):
        data = LaserScan()
        data.header.frame_id = "base_link"
        data.angle_min = 0
        data.angle_max = numpy.pi*2
        data.angle_increment = numpy.pi*2 / 200
        data.range_max = numpy.Inf
        data.range_min = 0
        theta = 0
        for i in range(200):
            r = 8.* numpy.sin(5. * theta )
            data.ranges.append(copy.deepcopy(r))
            data.intensities.append(theta)
            r = 8.* numpy.sin(5. * -theta)
            data.ranges.append(copy.deepcopy(r))
            data.intensities.append(theta)
 
            theta += data.angle_increment
        data.header.stamp = rospy.Time.now()
        self.pub_scan1.publish(data)
 
    def Timer_CB2(self, e):
        data = LaserScan()
        data.header.frame_id = "base_link"
        data.angle_min = 0
        data.angle_max = numpy.pi*2
        data.angle_increment = numpy.pi*2 / 200
        data.range_max = numpy.Inf
        data.range_min = 0
        theta = 0
        for i in range(200):
            r = 8. * numpy.cos(5. * theta) + 1
            data.intensities.append(theta)
            data.ranges.append(copy.deepcopy(r))
            r = 8. * numpy.cos(5. * -theta) + 1
            data.intensities.append(theta)
            data.ranges.append(copy.deepcopy(r))
            theta += data.angle_increment
 
        data.header.stamp = rospy.Time.now()
        self.pub_scan2.publish(data)
 
    def Timer_CB3(self, e):
        data = LaserScan()
        data.header.frame_id = "base_link"
        data.angle_min = 0
        data.angle_max = numpy.pi*2
        data.angle_increment = numpy.pi*2 / 50
        data.range_max = numpy.Inf
        data.range_min = 0
        theta = 0
        for i in range(200):
            r = 2. * numpy.sin(5. * theta) + 1
            data.intensities.append(theta)
            data.ranges.append(copy.deepcopy(r))
            r = 2. * numpy.sin(5. * -theta) + 1
            data.intensities.append(theta)
            data.ranges.append(copy.deepcopy(r))
            theta += data.angle_increment
 
        data.header.stamp = rospy.Time.now()
        self.pub_scan3.publish(data)
 
    #慎用！！！！
    def Timer_CB4(self, e):
        data = LaserScan()
        data.header.frame_id = "base_link"
        data.angle_min = 0
        data.angle_max = numpy.pi*2
        data.angle_increment = data.angle_max / 200
        data.range_max = numpy.Inf
        data.range_min = 0
        theta = 0
        for i in range(200):
            r = 9. * numpy.arccos(numpy.sin(theta)) + 9
            data.ranges.append(r)
            theta += data.angle_increment
 
        data.header.stamp = rospy.Time.now()
        self.pub_scan4.publish(data)
 
if __name__ == '__main__':
    node_name = 'Test_Maker'
    rospy.init_node(node_name)
    try:
        Test_Maker()
    except rospy.ROSInterruptException:
        rospy.logerr('%s error'%node_name)





