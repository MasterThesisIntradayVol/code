#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:18:43 2020

@author: tobbe
"""

# =============================================================================
# IMPORT MODULES
# =============================================================================

import yfinance as yf
import pandas as pd



# =============================================================================
# IMPORT DATA
# =============================================================================

msft = yf.Ticker('MSFT')
chain = msft.option_chain('2019-01-03')
