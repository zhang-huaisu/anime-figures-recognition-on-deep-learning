# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:41:59 2018

@author: 怀素
"""
import requests
from bs4 import BeautifulSoup
header = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0',
       'Referer': 'http://www.chuanxincao.net/touxiang/',
       'Cookie':'Hm_lvt_f1a7d3413c3e4fe2a405b56e7e6e5258=1524235135; Hm_lpvt_f1a7d3413c3e4fe2a405b56e7e6e5258=1524235727',
      }

def getPngtag():
    home_page ='http://www.chuanxincao.net/touxiang/'
    Png_requests = requests.get(home_page, headers = header)
    if Png_requests.status_code==200:
        print ("网页访问正常")
    Png_requests.encoding = Png_requests.apparent_encoding
    html_Png = Png_requests.text
    soup_Png = BeautifulSoup(html_Png, 'html.parser')  #创建一个实例。
    def has_src_but_no_border(soup_Png):
        return soup_Png.has_attr('src') and soup_Png.has_attr('alt') and not soup_Png.has_attr('border')
    tag_Png = soup_Png.find_all(has_src_but_no_border)
    print(tag_Png)
    return tag_Png
def getPng():
    for tag_Png in getPngtag():
        url_Png = tag_Png.attrs['src']  #筛选运算，保留含有src属性的标签。
        Png_name = tag_Png.attrs['alt']
        content_Png = requests.get(url_Png, headers = header).content
        name_Png = r'D:\\image\\' + Png_name +'.png'
        with open(name_Png, 'wb') as fp:
            fp.write(content_Png)
        print('已保存该图片',name_Png)
getPng()
