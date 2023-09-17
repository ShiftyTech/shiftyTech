from collections import deque

import pyautogui

from compiler.mapping import pose_mapping


class Compile:
    #last_operation: str = ''
    #writer: pyautogui
    #operations = deque()
    to_write = []

    def __init__(self, writer: pyautogui):
        self.writer = writer
        self.last_operation = ''
        self.operations = deque()

    def add_operation(self, pose: str):
        mapped_pose = pose_mapping[pose]
        print(mapped_pose)
        self.last_operation = mapped_pose
        self.operations.append(mapped_pose)
        self.last_operation = mapped_pose
        self.write_out()

    def write_out(self):
        pyautogui.write(self.last_operation)
