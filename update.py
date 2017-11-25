#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:01:13 2017

@author: eric.hensleyibm.com
"""

from database import line, moneyline, updatesql

updatesql.update_odds()
line.update_excel()
line.update_web_db()
moneyline.update_excel()
moneyline.update_web_db()