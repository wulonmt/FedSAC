import gymnasium as gym
from gymnasium.envs.mujoco.half_cheetah_v5 import HalfCheetahEnv
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import os
import time
from math import sin, cos, pi

class HalfCheetahFixLength(HalfCheetahEnv):
    def __init__(self, xml_file='./Env/envs/half_cheetah.xml', bthigh_scale = 1, fthigh_scale = 1, **kwrgs):
        self.xml_file = xml_file
        self.bthigh_scale = bthigh_scale
        self.fthigh_scale = fthigh_scale
        self.modified_xml = f"./Env/envs/half_cheetah_{self.bthigh_scale}_{self.fthigh_scale}.xml"
        
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
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        
        # Find the relevant elements
        bthigh = root.find(".//body[@name='bthigh']")
        bthigh_geom = bthigh.find(".//geom[@name='bthigh']")
        bshin = bthigh.find(".//body[@name='bshin']")
        
        fthigh = root.find(".//body[@name='fthigh']")
        fthigh_geom = fthigh.find(".//geom[@name='fthigh']")
        fshin = fthigh.find(".//body[@name='fshin']")
        
        torso = root.find(".//body[@name='torso']")
        
        # Get original lengths
        bthigh_ori_length = float(bthigh_geom.get('size').split()[1])
        fthigh_ori_length = float(fthigh_geom.get('size').split()[1])
        
        # Update back thigh geometry
        bthigh_radius = bthigh_geom.get('size').split()[0]
        new_bthigh_length = bthigh_ori_length * self.bthigh_scale
        bthigh_geom.set('size', f"{bthigh_radius} {new_bthigh_length}")
        
        # Calculate new positions for back thigh
        # Original angle is -3.8 radians
        angle_bthigh = -3.8 + pi
        # new_bthigh_pos_x = new_bthigh_length * cos(angle_bthigh)
        # new_bthigh_pos_z = new_bthigh_length * sin(angle_bthigh) - float(bthigh_radius)
        original_bthigh_pos = [float(x) for x in bthigh_geom.get('pos').split()]
        new_bthigh_pos_x = original_bthigh_pos[0] + cos(angle_bthigh) * (new_bthigh_length - bthigh_ori_length)
        new_bthigh_pos_z = original_bthigh_pos[2] + sin(angle_bthigh) * (new_bthigh_length - bthigh_ori_length)
        bthigh_geom.set('pos', f"{new_bthigh_pos_x} 0 {new_bthigh_pos_z}")
        
        # Update bshin position based on new bthigh length
        original_bshin_pos = [float(x) for x in bshin.get('pos').split()]
        new_bshin_x = original_bshin_pos[0] + 2 * cos(angle_bthigh) * (new_bthigh_length - bthigh_ori_length)
        new_bshin_z = original_bshin_pos[2] + 2 * sin(angle_bthigh) * (new_bthigh_length - bthigh_ori_length)
        bshin.set('pos', f"{new_bshin_x} 0 {new_bshin_z}")
        
        # Update front thigh geometry
        fthigh_radius = fthigh_geom.get('size').split()[0]
        new_fthigh_length = fthigh_ori_length * self.fthigh_scale
        fthigh_geom.set('size', f"{fthigh_radius} {new_fthigh_length}")
        
        # Calculate new positions for front thigh
        # Original angle is 0.52 radians
        angle_fthigh = 0.52
        new_fthigh_pos_x = -new_fthigh_length * sin(angle_fthigh)
        new_fthigh_pos_z = -new_fthigh_length * cos(angle_fthigh)
        fthigh_geom.set('pos', f"{new_fthigh_pos_x} 0 {new_fthigh_pos_z}")
        
        # Update fshin position based on new fthigh length
        original_fshin_pos = [float(x) for x in fshin.get('pos').split()]
        scale_factor_fthigh = self.fthigh_scale
        new_fshin_x = original_fshin_pos[0] * scale_factor_fthigh
        new_fshin_z = original_fshin_pos[2] * scale_factor_fthigh
        fshin.set('pos', f"{new_fshin_x} 0 {new_fshin_z}")
        
        # Update torso position if needed
        # This might need adjustment based on your specific requirements
        current_torso_z = float(torso.get('pos').split()[2])
        max_z = max(sin(angle_bthigh) * new_bthigh_length, sin(angle_fthigh) * new_fthigh_length)
        new_torso_z = current_torso_z + max_z
        torso.set('pos', f"0 0 {new_torso_z}")

        # 轉換為字符串並格式化
        rough_string = ET.tostring(root, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # 保存到文件
        with open(self.modified_xml, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)