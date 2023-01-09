import os
import platform

from matplotlib import pyplot as plt, font_manager


def os_env():
    if platform.system() == 'Linux':
        # 한글 폰트가 설치되어 있어야 함
        plt.rcParams['font.family'] = 'NanumGothic'
    elif platform.system() == 'Windows':
        # 한글이 깨지는 문제와 관련한 내용으로 vmoptions 뒤에
        # -Dfile.encoding=UTF-8
        # -Dconsole.encoding=UTF-8
        # 2가지를 같이 붙여줘야 한다
        os.system('chcp 65001')

        font_path = 'C:/Windows/Fonts/NGULIM.TTF'
        font = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font