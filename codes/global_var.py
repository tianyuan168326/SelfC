import os
import random
class GlobalVar:
    def __init__(self):
        GlobalVar.VIDEO_T_LEN = None
        GlobalVar.Istrain = None
        GlobalVar.encode_video_random_name = random.sample('zyxwvutsrqponmlkjihgfedcba',5)
    @staticmethod
    def get_Temporal_LEN():
        # exit(0)
        if not hasattr(GlobalVar,"VIDEO_T_LEN"):
            return None
        return GlobalVar.VIDEO_T_LEN
    @staticmethod
    def set_Temporal_LEN(v):
        # if not hasattr(GlobalVar,"VIDEO_T_LEN"):
        GlobalVar.VIDEO_T_LEN = v

    @staticmethod
    def get_Istrain():
        if not hasattr(GlobalVar,"Istrain"):
            return None
        return GlobalVar.Istrain
    @staticmethod
    def set_Istrain(v):
        GlobalVar.Istrain = v
    @staticmethod
    def get_v_random_name():
        if not hasattr(GlobalVar,"encode_video_random_name"):
            GlobalVar.encode_video_random_name =  "".join(random.sample('zyxwvutsrqponmlkjihgfedcba',5))
        return GlobalVar.encode_video_random_name