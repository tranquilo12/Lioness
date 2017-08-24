# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 12:08:03 2017

@author: Greg Sigalov <arrowstem@gmail.com>
         Arrow Science & Technology
         Champaign, Illinois, USA
"""

def get_time_hms(dt):
    sec_total = int(dt)
    seconds = int(sec_total % 60)
    minutes = int(((sec_total-seconds)/60)%60)
    hours = int((sec_total-60*minutes-seconds)/3600)
    return [hours,minutes,seconds]

def NiceTimeString(dt):
    [hours,minutes,seconds] = get_time_hms(dt)
    if hours > 10:
        s = str(hours) + ' hr'
    elif hours > 0:
        s = str(hours) + ' hr ' + str(minutes) + ' min'
    elif minutes > 10:
        s = str(minutes) + ' min'
    elif minutes > 0:
        s = str(minutes) + ' min ' + str(seconds) + ' sec'
    else:
        s = str(seconds) + ' sec'
    return s
