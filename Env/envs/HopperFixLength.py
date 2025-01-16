import gymnasium as gym
from gymnasium.envs.mujoco.hopper_v5 import HopperEnv
import numpy as np
from gymnasium.envs.registration import register
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import os
import time

class HopperFixLength(HopperEnv):
    def __init__(self, xml_file='./Env/envs/hopper.xml', thigh_scale = 1, leg_scale = 1, **kwrgs):
        self.xml_file = xml_file
        self.thigh_scale = thigh_scale
        self.leg_scale = leg_scale
        self.modified_xml = f"./Env/envs/hopper_{self.thigh_scale}_{self.leg_scale}.xml"
        
        # 先讀取和修改 XML，再調用父類的初始化
        if not os.path.exists(self.modified_xml):
            # 生成 XML 文件
            self._modify_xml()
            time.sleep(1)
        
        super().__init__(xml_file=self.modified_xml, **kwrgs)
    
    def _get_xml_string(self):
        # 返回修改後的 XML 字符串
        return self.modified_xml
    
    def _modify_xml(self):
        # 讀取 XML 文件
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        # Find the relevant elements
        thigh = root.find(".//body[@name='thigh']")
        thigh_geom = thigh.find(".//geom[@name='thigh_geom']")
        leg = thigh.find(".//body[@name='leg']")
        leg_geom = leg.find(".//geom[@name='leg_geom']")
        torso = root.find(".//body[@name='torso']")
        foot = leg.find(".//body[@name='foot']")
        
        # 1. Get original lengths
        thigh_ori_length = float(thigh_geom.get('size').split()[1])
        leg_ori_length = float(leg_geom.get('size').split()[1])
        
        # 2. Update torso position
        new_torso_y = 0.1 + 2 * thigh_ori_length * self.thigh_scale + 2 * leg_ori_length * self.leg_scale + 0.2
        torso.set('pos', f"0 0 {new_torso_y}")
        
        # 3. Update thigh geometry
        new_thigh_geom_pos = -thigh_ori_length * self.thigh_scale
        thigh_geom.set('pos', f"0 0 {new_thigh_geom_pos}")
        
        # Update thigh size (keep radius, modify length)
        thigh_radius = thigh_geom.get('size').split()[0]
        new_thigh_length = thigh_ori_length * self.thigh_scale
        thigh_geom.set('size', f"{thigh_radius} {new_thigh_length}")
        
        # 4. Update leg position
        new_leg_pos = -(2 * thigh_ori_length * self.thigh_scale + leg_ori_length * self.leg_scale)
        leg.set('pos', f"0 0 {new_leg_pos}")
        
        # Update leg size (keep radius, modify length)
        leg_radius = leg_geom.get('size').split()[0]
        new_leg_length = leg_ori_length * self.leg_scale
        leg_geom.set('size', f"{leg_radius} {new_leg_length}")
        
        # 5. Update joint positions
        # Update root joints (rootx, rooty, rootz)
        root_joints = torso.findall(".//joint[@pos='0 0 -1.25']")
        for joint in root_joints:
            joint.set('pos', f"0 0 -{new_torso_y}")
            
        # Update leg joint
        leg_joint = leg.find(".//joint[@name='leg_joint']")
        leg_joint.set('pos', f"0 0 {leg_ori_length * self.leg_scale}")

        # 6. Update foot position
        foot.set('pos', f"0.13 0 {-(0.1 + leg_ori_length * self.leg_scale)}")

        # 轉換為字符串並格式化
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # 保存到文件
        with open(self.modified_xml, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)