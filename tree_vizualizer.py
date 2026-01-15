from src.tree import Tree
from src.option_trade import OptionTrade
from src.market import Market
from src.trinomial_model import TrinomialModel
from PriceurBS import Call, Put
import time
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

# Paramètres du marché
underlying = 100.0
vol = 0.2
rate = 0.05
div_amount = 0.0  # cash dividend amount
ex_div_date = date(2026, 6, 13)
market = Market(underlying, vol, rate, div_amount, ex_div_date)

# Paramètres de l'arbre
nb_step = 20  # impossible d'aller au delà de 994 sinon erreur récursion
date_valuation = date.today()

# Paramètres de l'option
strike = 100.0
maturity = date(2026, 9, 24)
type_option = 'european'
direction = 'call'
option = OptionTrade(mat=maturity, call_put=direction, ex=type_option, k=strike)

# create tree and model (correct argument order)
tree = Tree(nb_step, market, option, date_valuation)
mod = TrinomialModel(date_valuation, tree)
print("Tree created with root underlying:", tree.root.underlying_i)

start = time.time()
tree.build_tree()
end = time.time()
print("temps d'exécution création arbre:", end - start, "secondes")

start = time.time()
print("L'option vaut = ", mod.price(option))
end = time.time()
print("temps d'exécution priceur récursif:", end - start, "secondes")

tree.plot_tree()